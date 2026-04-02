"""
cida/trainer.py - sklearn-style API for CIDA V8.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CIDAConfig
from .model import CIDAModel, CIDABertModel
from .cdp import cdp_loss


class CIDAClassifier:
    """
    sklearn-style wrapper around the CIDA V8 model.
    """

    def __init__(self, cfg: CIDAConfig):
        self.cfg     = cfg
        self._model  = None
        self.history = {}
        self._device = (torch.device('cuda') if torch.cuda.is_available()
                        else torch.device('cpu'))
        self.model_type = getattr(cfg, "model_type", "cida")
        self.bert_model_name = getattr(cfg, "bert_model_name", "bert-base-uncased")

    def _t(self, X):
        if isinstance(X, torch.Tensor):
            return X.long().to(self._device)
        return torch.tensor(np.array(X), dtype=torch.long).to(self._device)

    def _y(self, y):
        if isinstance(y, torch.Tensor):
            return y.long().to(self._device)
        return torch.tensor(np.array(y), dtype=torch.long).to(self._device)

    def _batches(self, X, y, shuffle=True):
        n   = len(X)
        idx = torch.randperm(n) if shuffle else torch.arange(n)
        bs  = self.cfg.batch_size
        for s in range(0, n, bs):
            b = idx[s:s + bs]
            yield X[b], y[b]

    def _accuracy(self, X_t, y_t, chunk=512):
        self._model.eval()
        correct = total = 0
        with torch.no_grad():
            for i in range(0, len(X_t), chunk):
                p = self._model(X_t[i:i + chunk]).argmax(-1)
                correct += (p == y_t[i:i + chunk]).sum().item()
                total   += len(y_t[i:i + chunk])
        return correct / total

    def param_count(self):
        if self._model is None:
            self._model = CIDAModel(self.cfg).to(self._device)
        return self._model.count_params()

    def param_breakdown(self):
        if self._model is None:
            self._model = CIDAModel(self.cfg).to(self._device)
        return self._model.param_breakdown()

    def fit(self, X, y, X_val=None, y_val=None):
        bs = self.cfg.batch_size
        while bs >= 16:
            try:
                self.cfg.batch_size = bs
                return self._fit_impl(X, y, X_val, y_val)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                bs //= 2
                print(f"\n  [OOM CATCH] VRAM exhausted. Restarting fit with batch_size={bs}...")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    bs //= 2
                    print(f"\n  [OOM CATCH] VRAM exhausted. Restarting fit with batch_size={bs}...")
                else:
                    raise e
        raise RuntimeError("OOM even with batch_size=16!")

    def _fit_impl(self, X, y, X_val=None, y_val=None):
        cfg = self.cfg
        # Создание модели
        if self.model_type == "cidabert":
            self._model = CIDABertModel(cfg, bert_model_name=self.bert_model_name).to(self._device)
        else:
            self._model = CIDAModel(cfg).to(self._device)
        model = self._model

        X_t = self._t(X);
        y_t = self._y(y)
        has_val = X_val is not None
        if has_val:
            X_v = self._t(X_val);
            y_v = self._y(y_val)

        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs, eta_min=1e-5)

        self.history = {k: [] for k in [
            'epoch', 'loss', 'loss_task', 'loss_conv', 'loss_progress', 'loss_div', 'loss_dom',
            'acc_train', 'acc_val', 'consensus_entropy', 'consensus_weights', 'lr']}

        best_acc, best_state, pat = 0.0, None, cfg.patience
        t0 = time.time()

        if cfg.verbose:
            dev = (torch.cuda.get_device_name(0) if self._device.type == 'cuda' else 'CPU')
            bd = model.param_breakdown()
            print(f"\n{'-' * 64}")
            print(
                f"  CIDA V8 ({'BERT+CDP' if self.model_type == 'cidabert' else 'CDP'})  |  {model.count_params():,} params")
            print(f"  device: {dev}")
            print(f"  encoder={bd['encoder']:,}  cdp={bd['cdp']:,}  base={bd['base']:,}")
            if self.model_type == 'cidabert':
                print(f"  bert_model={self.bert_model_name}")
            print(f"  agents={cfg.n_agents}  rounds={cfg.n_rounds}")
            print(f"  shared_round_weights={getattr(cfg, 'share_deliberation_layers', True)}")
            print(f"{'-' * 64}")

        amp_enabled = self._device.type == 'cuda'
        amp_device = 'cuda' if amp_enabled else 'cpu'
        scaler = torch.amp.GradScaler(amp_device, enabled=amp_enabled)

        for epoch in range(1, cfg.max_epochs + 1):
            model.train()
            ep_loss = ep_task = ep_conv = ep_progress = ep_div = ep_dom = 0.0
            nb = 0

            for xb, yb in self._batches(X_t, y_t):
                opt.zero_grad()
                with torch.amp.autocast(amp_device, enabled=amp_enabled):
                    logits, beliefs, cw, av = model.forward_full(xb)
                    loss, comp = cdp_loss(logits, yb, beliefs, av, cw, cfg)

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()

                ep_loss += loss.item()
                ep_task += comp['task']
                ep_conv += comp['conv']
                ep_progress += comp.get('progress', 0.0)
                ep_div += comp['div']
                ep_dom += comp['dom']
                nb += 1

            sched.step()

            # Metrics
            acc_tr = self._accuracy(X_t, y_t)
            acc_vl = self._accuracy(X_v, y_v) if has_val else None

            # Consensus analysis
            model.eval()
            with torch.no_grad():
                _, _, cw_sample, _ = model.forward_full(X_t[:256])
                cw_mean = cw_sample.mean(0).cpu().numpy()
                c_ent = float(-(cw_sample * (cw_sample + 1e-8).log()).sum(-1).mean())

            cur_lr = sched.get_last_lr()[0]

            self.history['epoch'].append(epoch)
            self.history['loss'].append(ep_loss / nb)
            self.history['loss_task'].append(ep_task / nb)
            self.history['loss_conv'].append(ep_conv / nb)
            self.history['loss_progress'].append(ep_progress / nb)
            self.history['loss_div'].append(ep_div / nb)
            self.history['loss_dom'].append(ep_dom / nb)
            self.history['acc_train'].append(acc_tr)
            self.history['acc_val'].append(acc_vl)
            self.history['consensus_entropy'].append(c_ent)
            self.history['consensus_weights'].append(cw_mean.tolist())
            self.history['lr'].append(cur_lr)

            # Early stopping
            monitor = acc_vl if has_val else acc_tr
            if monitor > best_acc + 1e-4:
                best_acc = monitor
                pat = cfg.patience
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                pat -= 1

            if cfg.verbose and (epoch % cfg.log_every == 0 or epoch == 1):
                w_str = ' '.join(f'{w:.3f}' for w in cw_mean)
                vs = f'  val={acc_vl:.4f}' if has_val else ''
                print(
                    f"  ep {epoch:3d}/{cfg.max_epochs}"
                    f"  loss={ep_loss / nb:.4f}"
                    f"(task={ep_task / nb:.3f} stab={ep_conv / nb:.3f} prog={ep_progress / nb:.3f}"
                    f" div={ep_div / nb:.3f} dom={ep_dom / nb:.3f})"
                    f"  tr={acc_tr:.4f}{vs}"
                    f"  w=[{w_str}]"
                    f"  {time.time() - t0:.0f}s")

            if pat <= 0:
                if cfg.verbose:
                    print(f"\n  Early stop ep {epoch}")
                break

        if best_state:
            model.load_state_dict(best_state)
        if cfg.verbose:
            print(f"\n  Done {time.time() - t0:.1f}s  |  best={best_acc:.4f}")
            print(f"{'-' * 64}\n")
        return self

    def predict(self, X):
        self._model.eval()
        X_t = self._t(X)
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_t), 512):
                preds.append(self._model(X_t[i:i+512]).argmax(-1).cpu().numpy())
        return np.concatenate(preds)

    def predict_proba(self, X):
        self._model.eval()
        X_t = self._t(X)
        probs = []
        with torch.no_grad():
            for i in range(0, len(X_t), 512):
                logits = self._model(X_t[i:i+512])
                probs.append(F.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(probs)

    def score(self, X, y):
        return float((self.predict(X) == np.array(y)).mean())

    def consensus_weights(self):
        """Средние веса консенсуса из последней эпохи."""
        if self.history.get('consensus_weights'):
            return self.history['consensus_weights'][-1]
        return None

    def save(self, path: str):
        torch.save({
            'state': self._model.state_dict(),
            'cfg':   self.cfg,
            'history': self.history,
        }, path)
        if self.cfg.verbose:
            print(f"  Saved: {path}  ({self.param_count():,} params)")

    @classmethod
    def load(cls, path: str):
        data = torch.load(path, map_location='cpu')
        obj  = cls(data['cfg'])
        obj._model = CIDAModel(data['cfg'])
        obj._model.load_state_dict(data['state'])
        obj._model.to(obj._device)
        obj.history = data.get('history', {})
        return obj

    def __repr__(self):
        return (f"CIDAClassifier(d={self.cfg.d_model}, "
                f"agents={self.cfg.n_agents}, rounds={self.cfg.n_rounds}, "
                f"params={self.param_count():,})")
