"""Linear decision heads on frozen NOMAD-EEG-Large features, their calibration, and metrics.

Head: the paper's probe (standardised features, class-balanced L2 logistic regression, C chosen
on validation from 1e-6 ... 1e3), fitted with full-batch L-BFGS in PyTorch so the 20,480-d
features of 100k+ windows fit in minutes on a GPU. sklearn's objective
0.5||w||^2 + C * sum_i w_i loss_i is used, divided by C*n.

Two outputs per window:
  probabilities  softmax(z / T + b): T and b fitted on validation (NLL), so a stated 0.9 is
                 right about 9 times in 10 at the validation set's class mix
  decision       argmax(z / T): the class-balanced choice (classes treated as equally likely),
                 which is what balanced accuracy, macro F1 and the paper's metrics score
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score, roc_auc_score)

C_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


class LinearHead:
    def __init__(self, mu, sd, W, b, classes, T=1.0, bias=None, C=None):
        self.mu, self.sd, self.W, self.b = mu, sd, W, b          # torch tensors (float32, cpu)
        self.classes, self.C = list(classes), C
        self.T = float(T)
        self.bias = torch.zeros(len(classes)) if bias is None else bias

    def logits(self, F_):
        F_ = torch.as_tensor(F_, dtype=torch.float32, device=self.W.device)
        return ((F_ - self.mu) / self.sd) @ self.W.T + self.b

    def outputs(self, F_):
        """(calibrated probabilities [N, K], balanced decisions [N], balanced probabilities [N, K])."""
        z = self.logits(F_)
        p = torch.softmax(z / self.T + self.bias, -1)
        q = torch.softmax(z / self.T, -1)
        return p.cpu().numpy(), q.argmax(-1).cpu().numpy(), q.cpu().numpy()

    def raw_probs(self, F_):
        return torch.softmax(self.logits(F_), -1).cpu().numpy()

    def to(self, dev):
        for k in ("mu", "sd", "W", "b", "bias"):
            setattr(self, k, getattr(self, k).to(dev))
        return self

    def state(self):
        return {k: (getattr(self, k).cpu() if torch.is_tensor(getattr(self, k)) else getattr(self, k))
                for k in ("mu", "sd", "W", "b", "classes", "T", "bias", "C")}

    @classmethod
    def from_state(cls, s):
        return cls(s["mu"], s["sd"], s["W"], s["b"], s["classes"], s["T"], s["bias"], s["C"])


# ------------------------------------------------------------------------- fitting
def standardise_stats(Ftr: torch.Tensor):
    mu = Ftr.mean(0)
    sd = Ftr.std(0) + 1e-8
    return mu, sd


def _fit_one(Z, y, w_cls, K, lam, W0=None, b0=None, max_iter=300):
    D = Z.shape[1]
    W = (W0.clone() if W0 is not None else torch.zeros(K, D, device=Z.device)).requires_grad_(True)
    b = (b0.clone() if b0 is not None else torch.zeros(K, device=Z.device)).requires_grad_(True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, history_size=20,
                            tolerance_grad=1e-7, tolerance_change=1e-10, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Z @ W.T + b, y, weight=w_cls) + 0.5 * lam * (W * W).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach(), b.detach()


def fit_head(Ftr, ytr, Fva, yva, classes, device="cuda", log=print):
    """Ftr/Fva: float tensors on `device`; returns (LinearHead, validation curve).

    Ftr is standardised IN PLACE (it is the largest tensor; a copy would not fit a 20 GB GPU
    for the seizure task) and must not be reused by the caller. Fva is left unchanged."""
    K, n = len(classes), len(ytr)
    mu, sd = standardise_stats(Ftr)
    Ztr = Ftr.sub_(mu).div_(sd)
    Zva = (Fva - mu) / sd
    ytr_t = torch.as_tensor(ytr, device=device)
    counts = torch.bincount(ytr_t, minlength=K).float()
    w_cls = n / (K * counts.clamp(min=1))                      # sklearn class_weight="balanced"
    key = "auroc" if K == 2 else "balanced_accuracy"
    curve, best, W, b = {}, None, None, None
    for C in sorted(C_GRID, reverse=True):                     # weak -> strong regularisation, warm start
        W, b = _fit_one(Ztr, ytr_t, w_cls, K, 1.0 / (C * n), W, b)
        z = Zva @ W.T + b
        p = torch.softmax(z, -1).cpu().numpy()
        v = (roc_auc_score(yva, p[:, 1]) if K == 2 else balanced_accuracy_score(yva, p.argmax(1)))
        curve[str(C)] = float(v)
        log(f"    C={C:<8g} val {key} {v:.4f}")
        if best is None or v > best[0]:
            best = (v, C, W.clone(), b.clone())
    v, C, W, b = best
    if C in (C_GRID[0], C_GRID[-1]):
        log(f"    [warn] best C={C} is at the edge of the grid")
    return LinearHead(mu, sd, W, b, classes, C=C), curve


def fit_calibration(z_val: torch.Tensor, y_val, iters=200):
    """Temperature T and per-class bias b minimising validation NLL of softmax(z/T + b)."""
    y = torch.as_tensor(y_val, device=z_val.device)
    logT = torch.zeros(1, device=z_val.device, requires_grad=True)
    b = torch.zeros(z_val.shape[1], device=z_val.device, requires_grad=True)
    opt = torch.optim.LBFGS([logT, b], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z_val / logT.exp() + b, y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(logT.detach().exp()), b.detach()


# ------------------------------------------------------------------------- metrics
def ece(y, p, bins=15):
    """Expected calibration error of the top-class confidence."""
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def brier(y, p):
    onehot = np.eye(p.shape[1])[y]
    return float(((p - onehot) ** 2).sum(1).mean())


def reliability(y, p, bins=10):
    """Binned confidence vs accuracy for plotting."""
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out.append({"bin": [float(lo), float(hi)], "n": int(m.sum()),
                        "confidence": float(conf[m].mean()), "accuracy": float((pred[m] == y[m]).mean())})
    return out


def scores(y, p, dec):
    """p: calibrated probabilities, dec: balanced decisions."""
    K = p.shape[1]
    s = {"balanced_accuracy": float(balanced_accuracy_score(y, dec)),
         "f1_macro": float(f1_score(y, dec, average="macro", zero_division=0)),
         "f1_weighted": float(f1_score(y, dec, average="weighted", zero_division=0)),
         "accuracy": float(accuracy_score(y, dec)),
         "ece": ece(y, p), "brier": brier(y, p)}
    if K == 2:
        two = len(np.unique(y)) == 2                      # AUROC is undefined with one class present
        s["auroc"] = float(roc_auc_score(y, p[:, 1])) if two else float("nan")
        s["pr_auc"] = float(average_precision_score(y, p[:, 1])) if two else float("nan")
    else:
        s["cohen_kappa"] = float(cohen_kappa_score(y, dec))
    return s


def bootstrap(y, p, dec, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    acc = {}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        for k, v in scores(y[i], p[i], dec[i]).items():
            acc.setdefault(k, []).append(v)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in acc.items()}


def by_subject(y, p, subjects):
    """Mean calibrated probability per subject; the subject's label is its (constant) window label."""
    ids = sorted(set(subjects))
    s = np.asarray(subjects)
    P = np.stack([p[s == i].mean(0) for i in ids])
    Y = np.array([np.bincount(y[s == i]).argmax() for i in ids])
    return Y, P
