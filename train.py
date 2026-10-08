"""Train the Siamese spectrum matcher on the public library -> outputs/pretrained.pt

Recipe follows lyn1874/raman_spectra_matching_with_contrastive_learning: class-balanced sampling,
positive pairs = two spectra (or two augmented views) of one substance, negative pairs = different
substances, binary cross-entropy on the pair logit. The augmentations additionally imitate a
handheld instrument (calibration shift, broader peaks, noise, residual baseline, shorter range) so a
model trained on lab spectra transfers to ours.
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import DATA_DIR, GRID, OUT_DIR
from model import Siamese

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def specimen_split(label, group, seed=0, max_test_classes=600):
    """Hold out one whole specimen for classes that have >= 2 specimens (unseen-specimen test)."""
    rng = np.random.default_rng(seed)
    test = np.zeros(len(label), dtype=bool)
    classes = [c for c in np.unique(label) if len(set(group[label == c])) >= 2]
    for c in rng.permutation(classes)[:max_test_classes]:
        test |= (label == c) & (group == rng.choice(sorted(set(group[label == c]))))
    return ~test, test


def augment(x, strength=1.0):
    """x: [B, W] on DEV, each spectrum scaled to max 1."""
    b, w = x.shape
    # peak broadening (instrument resolution)
    sig = torch.randint(0, 5, (1,)).item() * strength
    if sig > 0:
        k = torch.arange(-12, 13, device=x.device).float()
        k = torch.exp(-0.5 * (k / sig) ** 2)
        x = F.conv1d(x[:, None], (k / k.sum())[None, None], padding=12)[:, 0]
    # wavenumber calibration shift, up to +-6 cm^-1 per spectrum
    shift = torch.randint(-6, 7, (b, 1), device=x.device)
    idx = torch.arange(w, device=x.device)[None] + shift
    x = torch.gather(x, 1, idx.clamp(0, w - 1)) * ((idx >= 0) & (idx < w))
    # relative peak heights
    x = x.clamp_min(0) ** torch.empty(b, 1, device=x.device).uniform_(0.7, 1.4)
    # residual smooth baseline + noise
    t = torch.linspace(-1, 1, w, device=x.device)[None]
    coef = torch.randn(b, 3, device=x.device) * 0.05 * strength
    x = x + (coef[:, :1] + coef[:, 1:2] * t + coef[:, 2:] * t ** 2).abs()
    x = x + torch.randn_like(x) * torch.empty(b, 1, device=x.device).uniform_(0, 0.03) * strength
    # shorter measured range (most RRUFF spectra stop near 1300 cm^-1, ours do not)
    hi = torch.where(torch.rand(b, 1, device=x.device) < 0.3,
                     torch.randint(950, 1400, (b, 1), device=x.device), torch.full((b, 1), w, device=x.device))
    x = x * (torch.arange(w, device=x.device)[None] < hi)
    x = x.clamp_min(0)
    return x / x.amax(1, keepdim=True).clamp_min(1e-6)


class PairSampler:
    def __init__(self, x, label, seed=0):
        self.x = torch.as_tensor(np.asarray(x, dtype=np.float32))
        self.rng = np.random.default_rng(seed)
        self.classes, inv = np.unique(label, return_inverse=True)
        self.members = [np.flatnonzero(inv == i) for i in range(len(self.classes))]

    def batch(self, k):
        cls = self.rng.choice(len(self.classes), k, replace=False)
        a = [self.rng.choice(self.members[c]) for c in cls]
        b = [self.rng.choice(self.members[c]) for c in cls]
        return self.x[np.concatenate([a, b])], np.concatenate([cls, cls])


def train(samplers, steps, k=32, n_neg=3, lr=2e-4, init=None, log_every=250, mix=None):
    """samplers: list of PairSampler; mix: share of each batch drawn from each sampler."""
    model = Siamese(len(GRID)).to(DEV)
    if init is not None:
        model.load_state_dict(init)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
    mix = mix or [1 / len(samplers)] * len(samplers)
    ks = [max(2, round(k * m)) for m in mix]
    t0, run_loss, run_acc = time.time(), 0.0, 0.0
    for step in range(1, steps + 1):
        model.train()
        xs, cs, off = [], [], 0
        for s, kk in zip(samplers, ks):
            x, c = s.batch(min(kk, len(s.classes)))
            xs.append(x)
            cs.append(c + off)
            off += len(s.classes)
        x, cls = augment(torch.cat(xs).to(DEV)), torch.as_tensor(np.concatenate(cs), device=DEV)
        same = cls[:, None] == cls[None]
        with torch.no_grad():  # hardest negatives = most similar looking spectra of another substance
            xn = F.normalize(x, dim=1)
            sim = (xn @ xn.T).masked_fill(same, -1)
            hard = sim.topk(1, dim=1).indices
            rand = torch.multinomial((~same).float(), n_neg - 1)
        n = len(x)
        ia = torch.arange(n, device=DEV)
        pos_b = torch.multinomial((same & ~torch.eye(n, dtype=torch.bool, device=DEV)).float(), 1)[:, 0]
        a = torch.cat([ia, ia.repeat_interleave(n_neg)])
        b = torch.cat([pos_b, torch.cat([hard, rand], 1).flatten()])
        y = torch.cat([torch.ones(n), torch.zeros(n * n_neg)]).to(DEV)
        with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
            f = model.extractor(x[:, None])
            logit = model.score(f[a], f[b])
        loss = F.binary_cross_entropy_with_logits(logit.float(), y, pos_weight=torch.tensor(float(n_neg), device=DEV))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        run_loss += loss.item()
        run_acc += ((logit > 0).float() == y).float().mean().item()
        if step % log_every == 0 or step == steps:
            d = log_every if step % log_every == 0 else steps % log_every
            print(f"  step {step:5d}/{steps}  loss {run_loss / d:.4f}  pair-acc {run_acc / d:.3f}  "
                  f"{time.time() - t0:5.0f}s", flush=True)
            run_loss = run_acc = 0.0
    return model


def top_k_accuracy(model, xq, yq, xr, yr, ks=(1, 5)):
    """Match every query against the reference library; a hit = right substance among the k best."""
    order = np.argsort(-model.match(xq, xr).numpy(), axis=1)
    out = {}
    for k in ks:
        hits = []
        for i, o in enumerate(order):
            seen = list(dict.fromkeys(yr[o[:200]]))[:k]  # k best distinct substances
            hits.append(yq[i] in seen)
        out[k] = float(np.mean(hits))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--eval_queries", type=int, default=400)
    args = ap.parse_args()
    d = np.load(DATA_DIR / "public.npz")
    x, label, group = d["x"].astype(np.float32), d["label"], d["group"]
    tr, te = specimen_split(label, group)
    print(f"device {DEV} | public library: {len(x)} spectra, {len(np.unique(label))} classes | "
          f"train {tr.sum()} / held-out specimens {te.sum()}")
    torch.manual_seed(0)
    model = train([PairSampler(x[tr], label[tr])], args.steps)
    OUT_DIR.mkdir(exist_ok=True)
    torch.save({"state": model.state_dict(), "train_mask": tr, "steps": args.steps}, OUT_DIR / "pretrained.pt")
    q = np.random.default_rng(0).choice(np.flatnonzero(te), min(args.eval_queries, te.sum()), replace=False)
    acc = top_k_accuracy(model, x[q], label[q], x[tr], label[tr])
    print(f"public held-out specimens ({len(q)} queries vs {tr.sum()} references, "
          f"{len(np.unique(label[tr]))} classes): top-1 {acc[1]:.3f}  top-5 {acc[5]:.3f}")


if __name__ == "__main__":
    main()
