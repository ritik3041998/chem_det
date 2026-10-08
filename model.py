"""1D Xception encoder + Siamese matching head.

Ported from lyn1874/raman_spectra_matching_with_contrastive_learning (MIT licence),
model/model_inception.py - "Raman spectrum matching with contrastive representation learning",
Li et al., Analyst 2022. Same layers and sizes; the only change is that dropout is a real module,
so it switches off in eval() and predictions are deterministic.
"""
import torch
import torch.nn as nn


class SeparableConv1d(nn.Module):
    def __init__(self, cin, cout, kernel):
        super().__init__()
        self.depthwise = nn.Conv1d(cin, cin, kernel, padding=(kernel - 1) // 2, groups=cin, bias=False)
        self.pointwise = nn.Conv1d(cin, cout, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class XceptionBlock(nn.Module):
    def __init__(self, cin, cout, kernel, start_with_act):
        super().__init__()
        self.skip = None
        if cin != cout:
            self.skip = nn.Sequential(nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout))
        layers = []
        for i, c in enumerate([cin, cout]):
            if i > 0 or start_with_act:
                layers.append(nn.LeakyReLU(0.3))
            layers += [SeparableConv1d(c, cout, kernel), nn.BatchNorm1d(cout)]
        self.rep = nn.Sequential(*layers)

    def forward(self, x):
        return self.rep(x) + (x if self.skip is None else self.skip(x))


def _stem(cin, cout, kernel):
    layers = []
    for c in (cin, cout):
        layers += [nn.Conv1d(c, cout, kernel, padding=(kernel - 1) // 2), nn.BatchNorm1d(cout), nn.LeakyReLU(0.3)]
    return nn.Sequential(*layers)


class Xception(nn.Module):
    """[B, 1, W] spectrum -> [B, depth, W // 2] feature map."""

    def __init__(self, kernel=21, depth=128, stem_dim=64, dropout=0.5):
        super().__init__()
        self.stem1 = _stem(1, stem_dim // 2, kernel)
        self.pool = nn.MaxPool1d(2)
        self.stem2 = _stem(stem_dim // 2, stem_dim, kernel)
        self.block1 = XceptionBlock(stem_dim, depth, kernel, start_with_act=False)
        self.block2 = XceptionBlock(depth, depth, kernel, start_with_act=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.stem2(self.pool(self.stem1(x)))
        x = self.block1(self.drop(x))
        return self.block2(self.drop(x))


class Siamese(nn.Module):
    """Scores how likely two spectra are the same substance (logit; higher = more similar)."""

    def __init__(self, n_wave, kernel=21, depth=128, stem_dim=64, dropout=0.5):
        super().__init__()
        self.extractor = Xception(kernel, depth, stem_dim, dropout)
        self.agg_dot = nn.Conv1d(depth, 1, 1)
        self.agg_l1 = nn.Conv1d(depth, 1, 1)
        self.agg_wave = nn.Linear(n_wave // 2 * 2, 1)
        self.drop = nn.Dropout(dropout)
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.05)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def score(self, fa, fb):
        """fa, fb: broadcastable feature maps [..., depth, W//2] -> logits [...]."""
        shape = torch.broadcast_shapes(fa.shape, fb.shape)
        l1 = self.agg_l1(self.drop((fa - fb).abs()).reshape(-1, *shape[-2:])).squeeze(1)
        dot = self.agg_dot(self.drop(fa * fb).reshape(-1, *shape[-2:])).squeeze(1)
        return self.agg_wave(torch.cat([l1, dot], dim=-1)).reshape(shape[:-2])

    def forward(self, xa, xb):
        return self.score(self.extractor(xa), self.extractor(xb))

    @torch.no_grad()
    def match(self, xq, xr, ref_batch=256, q_batch=8):
        """Spectra xq [Q, W] vs reference spectra xr [R, W] -> similarity logits [Q, R] (CPU).

        Reference features are computed chunk by chunk so a library of any size fits in GPU memory.
        """
        self.eval()
        dev = next(self.parameters()).device
        xq, xr = torch.as_tensor(xq, dtype=torch.float32), torch.as_tensor(xr, dtype=torch.float32)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            fq = torch.cat([self.extractor(xq[i:i + 64, None].to(dev)) for i in range(0, len(xq), 64)])
            cols = []
            for j in range(0, len(xr), ref_batch):
                fr = self.extractor(xr[j:j + ref_batch, None].to(dev))
                cols.append(torch.cat([self.score(fq[i:i + q_batch, None], fr[None])
                                       for i in range(0, len(fq), q_batch)]).float().cpu())
        return torch.cat(cols, dim=1)
