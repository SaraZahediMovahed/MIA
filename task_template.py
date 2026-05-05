import os
import sys
import torch
import pandas as pd
import requests
import random
import argparse

from pathlib import Path
from torch.utils.data import Dataset
from torchvision.models import resnet18
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


# config
BASE = Path(__file__).parent
PUB_PATH = BASE / "pub.pt"
PRIV_PATH = BASE / "priv.pt"
MODEL_PATH = BASE / "model.pt"
OUTPUT_CSV = BASE / "submission.csv"

BASE_URL = "http://34.63.153.158"   #DONOT CHANGE

open_api_key = open("API_KEY.txt", "r").read().strip()
API_KEY = open_api_key
TASK_ID = "01-mia"  #DONOT CHANGE




# dataset classes
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]


# load datasets
print("Loading datasets...")
pub_ds = torch.load(PUB_PATH, weights_only=False)
priv_ds = torch.load(PRIV_PATH, weights_only=False)


# normalization (same as training)
MEAN = [0.7406, 0.5331, 0.7059]
STD = [0.1491, 0.1864, 0.1301]

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Normalize(mean=MEAN, std=STD),
])

pub_ds.transform = transform
priv_ds.transform = transform


# load model
print("Loading model...")
model = resnet18(weights=None)
model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
model.maxpool = torch.nn.Identity()
model.fc = torch.nn.Linear(512, 9)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()



_impl_key_file = BASE / "API_KEY.txt"
if _impl_key_file.exists():
    API_KEY = open(_impl_key_file, "r").read().strip()

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model = model.to(device)


# pub + priv pool
class CombinedDS(Dataset):
    def __init__(self, pub_ds, priv_ds):
        self.pub_ds = pub_ds
        self.priv_ds = priv_ds
        self.n_pub = len(pub_ds)
        self.n_priv = len(priv_ds)

    def __len__(self):
        return self.n_pub + self.n_priv

    def __getitem__(self, i):
        if i < self.n_pub:
            id_, img, label, *_ = self.pub_ds[i]
        else:
            id_, img, label, *_ = self.priv_ds[i - self.n_pub]
        return id_, img, label


# hyperparameters
N_SHADOW = 32
N_IN_PER_POINT = 6                              
N_OUT_PER_POINT = N_SHADOW - N_IN_PER_POINT     
SHADOW_EPOCHS = 60
SHADOW_WARMUP_EPOCHS = 5
SHADOW_LABEL_SMOOTHING = 0.08                  
N_AUG = 4 
N_PUB = len(pub_ds)
N_PRIV = len(priv_ds)
N_TOTAL = N_PUB + N_PRIV
combined_ds = CombinedDS(pub_ds, priv_ds)
print(f"N_PUB={N_PUB}  N_PRIV={N_PRIV}  N_TOTAL={N_TOTAL}")


def _id_to_str(t):
    if torch.is_tensor(t):
        try:
            return str(int(t.item()))
        except Exception:
            return str(t.tolist())
    return str(t)


def _aug_view(imgs, v):
    # deterministic test-time views
    if v == 0:
        return imgs
    if v == 1:
        return torch.flip(imgs, dims=[-1])
    if v == 2:
        return torch.flip(imgs, dims=[-2])
    if v == 3:
        return torch.flip(torch.flip(imgs, dims=[-1]), dims=[-2])
    raise ValueError(f"unknown aug view {v}")


def augment_batch_gpu(imgs, pad=4):
    # per-sample hflip + per-batch random crop with reflect padding (GPU)
    B = imgs.shape[0]
    flip_mask = (torch.rand(B, device=imgs.device) < 0.5).view(B, 1, 1, 1)
    imgs = torch.where(flip_mask, torch.flip(imgs, dims=[-1]), imgs)
    H, W = imgs.shape[-2], imgs.shape[-1]
    padded = F.pad(imgs, (pad, pad, pad, pad), mode="reflect")
    h_off = int(torch.randint(0, 2 * pad + 1, (1,)).item())
    w_off = int(torch.randint(0, 2 * pad + 1, (1,)).item())
    return padded[:, :, h_off:h_off + H, w_off:w_off + W]


class ShadowTrainSubset(Dataset):
    # returns (img, label); augmentation happens later on GPU
    def __init__(self, base_ds, indices):
        self.base_ds = base_ds
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        _, img, label, *_ = self.base_ds[self.indices[i]]
        return img, label


def make_resnet():
    m = resnet18(weights=None)
    m.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = torch.nn.Identity()
    m.fc = torch.nn.Linear(512, 9)
    return m


def train_shadow_model(train_indices, base_ds, epochs, device, label=""):
    shadow = make_resnet().to(device)
    loader = DataLoader(
        ShadowTrainSubset(base_ds, train_indices),
        batch_size=128,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    base_lr = 0.1
    optimizer = torch.optim.SGD(
        shadow.parameters(),
        lr=base_lr,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    cos_epochs = max(1, epochs - SHADOW_WARMUP_EPOCHS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cos_epochs, eta_min=1e-5
    )

    shadow.train()
    for epoch in range(epochs):
        if epoch < SHADOW_WARMUP_EPOCHS:
            wlr = base_lr * float(epoch + 1) / float(SHADOW_WARMUP_EPOCHS)
            for pg in optimizer.param_groups:
                pg["lr"] = wlr
        ep_loss, n_b = 0.0, 0
        ep_correct, n_seen = 0, 0
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            imgs = augment_batch_gpu(imgs)
            optimizer.zero_grad()
            logits = shadow(imgs)
            loss = F.cross_entropy(
                logits, labels, label_smoothing=SHADOW_LABEL_SMOOTHING
            )
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n_b += 1
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                ep_correct += (pred == labels).sum().item()
                n_seen += labels.size(0)
        if epoch >= SHADOW_WARMUP_EPOCHS:
            scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(
                f"  {label}epoch {epoch+1}/{epochs}  "
                f"train-loss={ep_loss / max(1, n_b):.4f}  "
                f"train-acc={ep_correct / max(1, n_seen):.4f}"
            )
    shadow.eval()
    return shadow


def get_aug_features(net, ds, indices, device, batch_size=512, n_aug=N_AUG):
    class _DS(Dataset):
        def __init__(self, ds, indices):
            self.ds = ds
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            id_, img, label, *_ = ds[self.indices[i]]
            return img, label, id_, self.indices[i]

    loader = DataLoader(
        _DS(ds, indices), batch_size=batch_size, shuffle=False, num_workers=0
    )
    res = {}
    net.eval()
    with torch.no_grad():
        for imgs, labels, ids_, idxs in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            B = imgs.shape[0]
            buf = [
                {
                    "id": _id_to_str(ids_[j]),
                    "label": int(labels[j].item()),
                    "loss": [],
                    "phi": [],
                    "conf": [],
                    "true_p": [],
                    "entropy": [],
                    "margin": [],
                }
                for j in range(B)
            ]
            for v in range(n_aug):
                logits = net(_aug_view(imgs, v))
                log_probs = F.log_softmax(logits, dim=1)
                log_p_y = log_probs.gather(1, labels.view(-1, 1)).squeeze(1)
                masked = log_probs.scatter(1, labels.view(-1, 1), float("-inf"))
                log_one_minus = torch.logsumexp(masked, dim=1)
                phi = log_p_y - log_one_minus
                losses = F.cross_entropy(logits, labels, reduction="none")
                probs = log_probs.exp()
                conf, _ = probs.max(dim=1)
                true_p = probs.gather(1, labels.view(-1, 1)).squeeze(1)
                entropy = -(probs * log_probs).sum(dim=1)
                top2v = torch.topk(probs, k=min(2, probs.shape[1]), dim=1).values
                margin = top2v[:, 0] - top2v[:, 1] if top2v.shape[1] >= 2 else top2v[:, 0]
                for j in range(B):
                    buf[j]["loss"].append(float(losses[j].item()))
                    buf[j]["phi"].append(float(phi[j].item()))
                    buf[j]["conf"].append(float(conf[j].item()))
                    buf[j]["true_p"].append(float(true_p[j].item()))
                    buf[j]["entropy"].append(float(entropy[j].item()))
                    buf[j]["margin"].append(float(margin[j].item()))
            for j in range(B):
                res[idxs[j].item()] = buf[j]
    return res


def _eval(name, score, y):
    auc = roc_auc_score(y, score)
    fpr, tpr, _ = roc_curve(y, score)
    tpr5 = float(tpr[np.searchsorted(fpr, 0.05)])
    print(f"  {name:>22s}  AUC={auc:.4f}  TPR@5%FPR={tpr5:.4f}")
    return tpr5


def _tpr5(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[np.searchsorted(fpr, 0.05)])


def _rank01(x):
    r = np.argsort(np.argsort(x))
    return r / max(1, len(r) - 1)


# shadow membership matrix: in_matrix[s, ci] = True iff shadow s trains on ci
# every example gets exactly N_IN_PER_POINT IN shadows and N_OUT_PER_POINT OUT
rng = np.random.RandomState(42)
in_matrix = np.zeros((N_SHADOW, N_TOTAL), dtype=bool)
for j in range(N_TOTAL):
    chosen = rng.choice(N_SHADOW, size=N_IN_PER_POINT, replace=False)
    in_matrix[chosen, j] = True
assert (in_matrix.sum(axis=0) == N_IN_PER_POINT).all(), "Per-point IN balance failed"

print(
    f"Each example: {N_IN_PER_POINT} shadows IN, {N_OUT_PER_POINT} shadows OUT."
)
print(
    f"Per-shadow training-set sizes: "
    f"min={in_matrix.sum(axis=1).min()}  "
    f"max={in_matrix.sum(axis=1).max()}  "
    f"mean={in_matrix.sum(axis=1).mean():.0f}"
)


# target features + pub diagnostics
print("=" * 60)
print("Pre-shadow diagnostics on pub set")
print("=" * 60)

pub_y = np.array([int(pub_ds[i][3]) for i in range(N_PUB)], dtype=np.int64)
print(
    f"pub_y distribution: 0 -> {(pub_y == 0).sum()}  |  "
    f"1 -> {(pub_y == 1).sum()}  |  total {N_PUB}"
)

# single pass over combined pool; target_combined[ci] = dict for combined idx ci
target_combined = get_aug_features(model, combined_ds, list(range(N_TOTAL)),
                                    device, batch_size=512)

t_loss_orig = np.array([target_combined[i]["loss"][0] for i in range(N_PUB)])
t_phi_orig = np.array([target_combined[i]["phi"][0] for i in range(N_PUB)])
t_loss_avg = np.array([np.mean(target_combined[i]["loss"]) for i in range(N_PUB)])
t_phi_avg = np.array([np.mean(target_combined[i]["phi"]) for i in range(N_PUB)])
t_loss_min = np.array([min(target_combined[i]["loss"]) for i in range(N_PUB)])
t_phi_max = np.array([max(target_combined[i]["phi"]) for i in range(N_PUB)])
t_conf_orig = np.array([target_combined[i]["conf"][0] for i in range(N_PUB)])

print(
    f"target-loss summary: min={t_loss_orig.min():.4f}  "
    f"median={np.median(t_loss_orig):.4f}  mean={t_loss_orig.mean():.4f}  "
    f"max={t_loss_orig.max():.4f}"
)
if (pub_y == 1).sum() > 0 and (pub_y == 0).sum() > 0:
    m_loss_in = t_loss_orig[pub_y == 1].mean()
    m_loss_out = t_loss_orig[pub_y == 0].mean()
    m_phi_in = t_phi_orig[pub_y == 1].mean()
    m_phi_out = t_phi_orig[pub_y == 0].mean()
    print(
        f"target-loss mean | members={m_loss_in:.5f}  "
        f"non-members={m_loss_out:.5f}  diff={m_loss_in - m_loss_out:+.5f}"
    )
    print(
        f"target-phi  mean | members={m_phi_in:.5f}  "
        f"non-members={m_phi_out:.5f}  diff={m_phi_in - m_phi_out:+.5f}"
    )

_eval("-loss(orig)", -t_loss_orig, pub_y)
_eval("phi(orig)", t_phi_orig, pub_y)
_eval("-loss(avg-aug)", -t_loss_avg, pub_y)
_eval("phi(avg-aug)", t_phi_avg, pub_y)
_eval("-loss(min-aug)", -t_loss_min, pub_y)
_eval("phi(max-aug)", t_phi_max, pub_y)
_eval("conf(orig)", t_conf_orig, pub_y)
print("=" * 60)


# train shadows and query each on the full combined pool
print(
    f"\nTraining {N_SHADOW} shadows on combined pool "
    f"({SHADOW_EPOCHS} epochs, warmup={SHADOW_WARMUP_EPOCHS}, crop+flip)..."
)
shadow_combined = [None] * N_SHADOW

for s in range(N_SHADOW):
    train_idx = np.where(in_matrix[s])[0].tolist()
    print(f"Shadow {s+1}/{N_SHADOW}  train_size={len(train_idx)}")
    sh = train_shadow_model(train_idx, combined_ds, epochs=SHADOW_EPOCHS, device=device)
    shadow_combined[s] = get_aug_features(sh, combined_ds, list(range(N_TOTAL)),
                                           device, batch_size=512)
    del sh
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _info_arrs(info):
    return {
        "loss":    np.asarray(info["loss"], dtype=np.float64),
        "phi":     np.asarray(info["phi"], dtype=np.float64),
        "conf":    np.asarray(info["conf"], dtype=np.float64),
        "true_p":  np.asarray(info["true_p"], dtype=np.float64),
        "entropy": np.asarray(info["entropy"], dtype=np.float64),
        "margin":  np.asarray(info["margin"], dtype=np.float64),
    }


def _safe_logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p) - np.log1p(-p)


def _robust_stats(arr):
    # median + MAD (scaled to match a Gaussian std)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) * 1.4826
    return med, mad + 1e-6


def _trim_mean_std(a, prop=0.1):
    """Robust mean/std over OUT-shadow scalars (reduces one-off bad shadows)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    n = a.size
    if n <= 4:
        return float(a.mean()), float(a.std(ddof=0) + 1e-6)
    lo, hi = np.quantile(a, [prop, 1.0 - prop])
    mid = a[(a >= lo) & (a <= hi)]
    if mid.size < max(4, n // 4):
        mid = a
    return float(mid.mean()), float(mid.std(ddof=0) + 1e-6)


def build_features(target_combined, shadow_combined, in_matrix, indices,
                   class_phi_mean, class_loss_mean):

    feats = []
    for ci in indices:
        ti = _info_arrs(target_combined[ci])
        label = target_combined[ci]["label"]
        in_s = [s for s in range(N_SHADOW) if in_matrix[s, ci]]
        out_s = [s for s in range(N_SHADOW) if not in_matrix[s, ci]]
        f = []

        # 1) raw target features (all deterministic aug views + primary entropy/margin)
        for vi in range(N_AUG):
            f.append(ti["loss"][vi])
        for vi in range(N_AUG):
            f.append(ti["phi"][vi])
        for vi in range(N_AUG):
            f.append(ti["conf"][vi])
        for vi in range(N_AUG):
            f.append(ti["true_p"][vi])
        f.extend([ti["entropy"][0], ti["margin"][0]])

        # 2) aggregated target features
        f.extend([
            ti["loss"].mean(), ti["loss"].min(), ti["loss"].max(),
            ti["phi"].mean(),  ti["phi"].min(),  ti["phi"].max(),
            ti["conf"].mean(),
            ti["true_p"].mean(),
            ti["entropy"].mean(), ti["margin"].mean(),
            np.log(ti["loss"].mean() + 1e-12),
            np.log(ti["loss"].min() + 1e-12),
            _safe_logit(ti["true_p"].mean()),
            ti["loss"].max() - ti["loss"].min(),
            ti["phi"].max()  - ti["phi"].min(),
        ])

        # 3) OUT shadow distribution stats
        s_loss_mean = np.array([np.mean(shadow_combined[s][ci]["loss"]) for s in out_s])
        s_loss_min  = np.array([min(shadow_combined[s][ci]["loss"])     for s in out_s])
        s_phi_mean  = np.array([np.mean(shadow_combined[s][ci]["phi"])  for s in out_s])
        s_phi_max   = np.array([max(shadow_combined[s][ci]["phi"])      for s in out_s])
        s_truep_mean = np.array([np.mean(shadow_combined[s][ci]["true_p"]) for s in out_s])
        s_conf_mean = np.array([np.mean(shadow_combined[s][ci]["conf"]) for s in out_s])

        mu_loss, sd_loss = _trim_mean_std(s_loss_mean)
        mu_loss_min, sd_loss_min = _trim_mean_std(s_loss_min)
        mu_phi, sd_phi = _trim_mean_std(s_phi_mean)
        mu_phi_max, sd_phi_max = _trim_mean_std(s_phi_max)
        mu_truep = s_truep_mean.mean()
        mu_conf = s_conf_mean.mean()

        # 4) offline LiRA z-scores (high z => more likely member)
        z_loss     = (mu_loss     - ti["loss"].mean()) / sd_loss
        z_loss_min = (mu_loss_min - ti["loss"].min())  / sd_loss_min
        z_phi      = (ti["phi"].mean() - mu_phi)       / sd_phi
        z_phi_max  = (ti["phi"].max()  - mu_phi_max)   / sd_phi_max

        f.extend([
            mu_loss, sd_loss, mu_loss_min, sd_loss_min,
            mu_phi,  sd_phi,  mu_phi_max,  sd_phi_max,
            mu_truep, mu_conf,
            z_loss, z_loss_min, z_phi, z_phi_max,
            mu_loss     - ti["loss"].mean(),
            mu_loss_min - ti["loss"].min(),
            ti["phi"].mean() - mu_phi,
            ti["phi"].max()  - mu_phi_max,
            ti["true_p"].mean() - mu_truep,
            ti["conf"].mean()   - mu_conf,
        ])

        # 5) robust OUT statistics (median + MAD), immune to bad shadows
        med_loss, mad_loss = _robust_stats(s_loss_mean)
        med_phi,  mad_phi  = _robust_stats(s_phi_mean)
        med_loss_min, mad_loss_min = _robust_stats(s_loss_min)
        med_phi_max,  mad_phi_max  = _robust_stats(s_phi_max)
        z_loss_robust     = (med_loss     - ti["loss"].mean()) / mad_loss
        z_loss_min_robust = (med_loss_min - ti["loss"].min())  / mad_loss_min
        z_phi_robust      = (ti["phi"].mean() - med_phi)       / mad_phi
        z_phi_max_robust  = (ti["phi"].max()  - med_phi_max)   / mad_phi_max
        f.extend([
            med_loss, mad_loss, med_phi, mad_phi,
            z_loss_robust, z_loss_min_robust, z_phi_robust, z_phi_max_robust,
        ])

        # 6) per-class normalisation (uses only classification labels)
        c_phi = float(class_phi_mean.get(label, 0.0))
        c_loss = float(class_loss_mean.get(label, 0.0))
        f.extend([
            c_phi, c_loss,
            ti["phi"].mean() - c_phi,
            ti["phi"].max()  - c_phi,
            ti["loss"].mean() - c_loss,
            ti["loss"].min()  - c_loss,
            (ti["phi"].mean() - c_phi) - (mu_phi - c_phi),
            (mu_loss - c_loss) - (ti["loss"].mean() - c_loss),
        ])

        # 7) IN-distribution + online-LiRA features
        i_loss_mean = np.array([np.mean(shadow_combined[s][ci]["loss"]) for s in in_s])
        i_phi_mean  = np.array([np.mean(shadow_combined[s][ci]["phi"])  for s in in_s])
        mu_loss_in, sd_loss_in = i_loss_mean.mean(), i_loss_mean.std() + 1e-6
        mu_phi_in,  sd_phi_in  = i_phi_mean.mean(),  i_phi_mean.std()  + 1e-6

        # floor std estimates (N_IN_PER_POINT IN shadows — stabilize tiny-batch variance)
        sd_loss_in_c = max(sd_loss_in, 0.05)
        sd_phi_in_c  = max(sd_phi_in,  0.10)
        sd_loss_out_c = max(sd_loss,   0.05)
        sd_phi_out_c  = max(sd_phi,    0.10)
        # Shrink IN std toward OUT (small-N IN moments are noisy; stabilises online LLR)
        _sh = 0.26
        sd_loss_in_c = (1.0 - _sh) * sd_loss_in_c + _sh * sd_loss_out_c
        sd_phi_in_c = (1.0 - _sh) * sd_phi_in_c + _sh * sd_phi_out_c

        target_loss_avg = float(ti["loss"].mean())
        target_phi_avg  = float(ti["phi"].mean())

        # log N(target | IN) - log N(target | OUT)
        log_p_loss_in  = -0.5 * ((target_loss_avg - mu_loss_in)  / sd_loss_in_c) ** 2 - np.log(sd_loss_in_c)
        log_p_loss_out = -0.5 * ((target_loss_avg - mu_loss)     / sd_loss_out_c) ** 2 - np.log(sd_loss_out_c)
        online_loss = log_p_loss_in - log_p_loss_out

        log_p_phi_in  = -0.5 * ((target_phi_avg - mu_phi_in) / sd_phi_in_c) ** 2 - np.log(sd_phi_in_c)
        log_p_phi_out = -0.5 * ((target_phi_avg - mu_phi)    / sd_phi_out_c) ** 2 - np.log(sd_phi_out_c)
        online_phi = log_p_phi_in - log_p_phi_out

        # shared-sigma_OUT "simple" online z-score: robust to noisy sigma_IN
        online_phi_simple  = (mu_phi_in  - mu_phi)  * (target_phi_avg  - 0.5 * (mu_phi_in  + mu_phi))  / (sd_phi_out_c  ** 2)
        online_loss_simple = (mu_loss    - mu_loss_in) * (mu_loss + mu_loss_in - 2 * target_loss_avg) / (sd_loss_out_c ** 2)

        f.extend([
            mu_loss_in, sd_loss_in, mu_phi_in, sd_phi_in,
            target_loss_avg - mu_loss_in,
            target_phi_avg  - mu_phi_in,
            mu_loss_in - mu_loss,
            mu_phi_in  - mu_phi,
            online_loss, online_phi,
            online_loss_simple, online_phi_simple,
        ])

        feats.append(np.asarray(f, dtype=np.float64))
    return np.stack(feats, axis=0)


print("\nBuilding feature matrices...")


class_phi_mean = {}
class_loss_mean = {}
for c in range(9):
    phi_vals = []
    loss_vals = []
    for ci in range(N_TOTAL):
        if target_combined[ci]["label"] == c:
            phi_vals.append(np.mean(target_combined[ci]["phi"]))
            loss_vals.append(np.mean(target_combined[ci]["loss"]))
    if phi_vals:
        class_phi_mean[c] = float(np.mean(phi_vals))
        class_loss_mean[c] = float(np.mean(loss_vals))
    else:
        class_phi_mean[c] = 0.0
        class_loss_mean[c] = 0.0
print("Per-class target mean phi:", {c: round(v, 3) for c, v in class_phi_mean.items()})

pub_X = build_features(target_combined, shadow_combined, in_matrix,
                       indices=list(range(N_PUB)),
                       class_phi_mean=class_phi_mean,
                       class_loss_mean=class_loss_mean)
priv_X = build_features(target_combined, shadow_combined, in_matrix,
                        indices=list(range(N_PUB, N_TOTAL)),
                        class_phi_mean=class_phi_mean,
                        class_loss_mean=class_loss_mean)
pub_X = np.nan_to_num(pub_X, nan=0.0, posinf=1e6, neginf=-1e6)
priv_X = np.nan_to_num(priv_X, nan=0.0, posinf=1e6, neginf=-1e6)
print(f"Feature shapes:  pub_X={pub_X.shape}  priv_X={priv_X.shape}")

# feature column indices
N_TARGET_RAW = 4 * N_AUG + 2
N_TARGET_AGG = 15  
N_OUT_BASIC = 10    
N_Z_RAWDIFF = 10    
N_ROBUST_BASIC = 4
N_ROBUST_Z = 4
N_PERCLASS = 8

Z_BASE = N_TARGET_RAW + N_TARGET_AGG + N_OUT_BASIC
Z_LOSS = Z_BASE + 0
Z_LOSS_MIN = Z_BASE + 1
Z_PHI = Z_BASE + 2
Z_PHI_MAX = Z_BASE + 3

ROBUST_Z_BASE = Z_BASE + N_Z_RAWDIFF + N_ROBUST_BASIC
Z_LOSS_R = ROBUST_Z_BASE + 0
Z_LOSS_MIN_R = ROBUST_Z_BASE + 1
Z_PHI_R = ROBUST_Z_BASE + 2
Z_PHI_MAX_R = ROBUST_Z_BASE + 3

ONLINE_BASE = ROBUST_Z_BASE + N_ROBUST_Z + N_PERCLASS
ONLINE_LL_LOSS = ONLINE_BASE + 8
ONLINE_LL_PHI = ONLINE_BASE + 9
ONLINE_SIMPLE_LOSS = ONLINE_BASE + 10
ONLINE_SIMPLE_PHI = ONLINE_BASE + 11

_idx0 = 0
_ti0 = _info_arrs(target_combined[_idx0])
_out_s0 = [s for s in range(N_SHADOW) if not in_matrix[s, _idx0]]
_in_s0  = [s for s in range(N_SHADOW) if in_matrix[s, _idx0]]
_s_phi_mean0 = np.array([np.mean(shadow_combined[s][_idx0]["phi"]) for s in _out_s0])
_i_phi_mean0 = np.array([np.mean(shadow_combined[s][_idx0]["phi"]) for s in _in_s0])
_mu_phi0, _sd_phi0 = _trim_mean_std(_s_phi_mean0)
_z_phi0 = (_ti0["phi"].mean() - _mu_phi0) / _sd_phi0
assert abs(pub_X[_idx0, Z_PHI] - _z_phi0) < 1e-6, (
    f"Feature index Z_PHI wrong: got pub_X[0, Z_PHI]={pub_X[_idx0, Z_PHI]}, "
    f"expected {_z_phi0}"
)
_med_phi0, _mad_phi0 = _robust_stats(_s_phi_mean0)
_z_phi_r0 = (_ti0["phi"].mean() - _med_phi0) / _mad_phi0
assert abs(pub_X[_idx0, Z_PHI_R] - _z_phi_r0) < 1e-6, (
    f"Feature index Z_PHI_R wrong: got pub_X[0, Z_PHI_R]={pub_X[_idx0, Z_PHI_R]}, "
    f"expected {_z_phi_r0}"
)
_mu_phi_in0 = _i_phi_mean0.mean()
_sd_phi_out0_c = max(_sd_phi0, 0.10)
_online_simple_phi0 = (_mu_phi_in0 - _mu_phi0) * (_ti0["phi"].mean() - 0.5 * (_mu_phi_in0 + _mu_phi0)) / (_sd_phi_out0_c ** 2)
assert abs(pub_X[_idx0, ONLINE_SIMPLE_PHI] - _online_simple_phi0) < 1e-6, (
    f"Feature index ONLINE_SIMPLE_PHI wrong: got pub_X[0, ONLINE_SIMPLE_PHI]={pub_X[_idx0, ONLINE_SIMPLE_PHI]}, "
    f"expected {_online_simple_phi0}"
)

print("=" * 60)
print("LiRA z-score & online-LiRA baselines on pub")
print("=" * 60)
for name, idx in [
    ("z_loss", Z_LOSS),
    ("z_loss_min", Z_LOSS_MIN),
    ("z_phi", Z_PHI),
    ("z_phi_max", Z_PHI_MAX),
    ("z_loss_robust", Z_LOSS_R),
    ("z_loss_min_robust", Z_LOSS_MIN_R),
    ("z_phi_robust", Z_PHI_R),
    ("z_phi_max_robust", Z_PHI_MAX_R),
    ("online_loss_LL", ONLINE_LL_LOSS),
    ("online_phi_LL", ONLINE_LL_PHI),
    ("online_loss_simple", ONLINE_SIMPLE_LOSS),
    ("online_phi_simple", ONLINE_SIMPLE_PHI),
]:
    _eval(name, pub_X[:, idx], pub_y)


print("=" * 60)
print("Training attack classifiers with 5-fold CV")
print("=" * 60)

CORE_COLS = [
    Z_LOSS, Z_LOSS_MIN, Z_PHI, Z_PHI_MAX,
    Z_LOSS_R, Z_LOSS_MIN_R, Z_PHI_R, Z_PHI_MAX_R,
    ONLINE_LL_LOSS, ONLINE_LL_PHI, ONLINE_SIMPLE_LOSS, ONLINE_SIMPLE_PHI,
]
print(f"core feature columns ({len(CORE_COLS)}): {CORE_COLS}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
oof_lr_full = np.zeros(N_PUB, dtype=np.float64)
oof_lr_core = np.zeros(N_PUB, dtype=np.float64)
oof_mlp     = np.zeros(N_PUB, dtype=np.float64)
priv_lr_full = np.zeros(N_PRIV, dtype=np.float64)
priv_lr_core = np.zeros(N_PRIV, dtype=np.float64)
priv_mlp     = np.zeros(N_PRIV, dtype=np.float64)

for fold, (tr, va) in enumerate(skf.split(pub_X, pub_y)):
    
    sc_full = StandardScaler()
    Xtr_f = sc_full.fit_transform(pub_X[tr])
    Xva_f = sc_full.transform(pub_X[va])
    Xpv_f = sc_full.transform(priv_X)
    lr_f = LogisticRegression(C=0.3, max_iter=3000, solver="lbfgs")
    lr_f.fit(Xtr_f, pub_y[tr])
    oof_lr_full[va] = lr_f.predict_proba(Xva_f)[:, 1]
    priv_lr_full += lr_f.predict_proba(Xpv_f)[:, 1]

    
    sc_core = StandardScaler()
    Xtr_c = sc_core.fit_transform(pub_X[tr][:, CORE_COLS])
    Xva_c = sc_core.transform(pub_X[va][:, CORE_COLS])
    Xpv_c = sc_core.transform(priv_X[:, CORE_COLS])
    lr_c = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")
    lr_c.fit(Xtr_c, pub_y[tr])
    oof_lr_core[va] = lr_c.predict_proba(Xva_c)[:, 1]
    priv_lr_core += lr_c.predict_proba(Xpv_c)[:, 1]

    
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        random_state=fold,
    )
    mlp.fit(Xtr_f, pub_y[tr])
    oof_mlp[va] = mlp.predict_proba(Xva_f)[:, 1]
    priv_mlp += mlp.predict_proba(Xpv_f)[:, 1]

priv_lr_full /= skf.get_n_splits()
priv_lr_core /= skf.get_n_splits()
priv_mlp     /= skf.get_n_splits()

oof_lr_blend = 0.5 * _rank01(oof_lr_full) + 0.5 * _rank01(oof_lr_core)
priv_lr_blend = 0.5 * _rank01(priv_lr_full) + 0.5 * _rank01(priv_lr_core)

t_lr_full = _eval("OOF-LR-FULL", oof_lr_full, pub_y)
t_lr_core = _eval("OOF-LR-CORE", oof_lr_core, pub_y)
t_lr_blend = _eval("OOF-LR-BLEND", oof_lr_blend, pub_y)
t_mlp     = _eval("OOF-MLP",      oof_mlp,     pub_y)



print("Training per-class LR attacker (9 classes) ...")
pub_class = np.array([target_combined[i]["label"] for i in range(N_PUB)])
priv_class = np.array([target_combined[N_PUB + i]["label"] for i in range(N_PRIV)])
oof_lr_perclass  = np.zeros(N_PUB, dtype=np.float64)
priv_lr_perclass = np.zeros(N_PRIV, dtype=np.float64)
priv_fold_counts = np.zeros(N_PRIV, dtype=np.float64)

for fold, (tr, va) in enumerate(skf.split(pub_X, pub_y)):
    for c in range(9):
        tr_c = tr[pub_class[tr] == c]
        va_c = va[pub_class[va] == c]
        priv_mask_c = (priv_class == c)
        if tr_c.size < 20 or va_c.size == 0:
            continue
        if len(np.unique(pub_y[tr_c])) < 2:
            continue
        sc_pc = StandardScaler()
        Xtr_pc = sc_pc.fit_transform(pub_X[tr_c])
        Xva_pc = sc_pc.transform(pub_X[va_c])
        lr_pc = LogisticRegression(C=0.5, max_iter=3000, solver="lbfgs")
        lr_pc.fit(Xtr_pc, pub_y[tr_c])
        oof_lr_perclass[va_c] = lr_pc.predict_proba(Xva_pc)[:, 1]
        if priv_mask_c.any():
            Xpv_pc = sc_pc.transform(priv_X[priv_mask_c])
            priv_lr_perclass[priv_mask_c] += lr_pc.predict_proba(Xpv_pc)[:, 1]
            priv_fold_counts[priv_mask_c] += 1.0

priv_lr_perclass = np.where(
    priv_fold_counts > 0,
    priv_lr_perclass / np.maximum(priv_fold_counts, 1.0),
    priv_lr_full,
)

t_lr_perclass = _eval("OOF-LR-PERCLASS", oof_lr_perclass, pub_y)



oof_super = (
    _rank01(oof_lr_full)
    + _rank01(oof_lr_core)
    + _rank01(oof_mlp)
    + _rank01(oof_lr_perclass)
) / 4.0
priv_super = (
    _rank01(priv_lr_full)
    + _rank01(priv_lr_core)
    + _rank01(priv_mlp)
    + _rank01(priv_lr_perclass)
) / 4.0
t_super = _eval("OOF-SUPER-BLEND", oof_super, pub_y)

oof_blend3 = (
    _rank01(oof_lr_full) + _rank01(oof_lr_core) + _rank01(oof_mlp)
) / 3.0
priv_blend3 = (
    _rank01(priv_lr_full) + _rank01(priv_lr_core) + _rank01(priv_mlp)
) / 3.0
t_blend3 = _eval("OOF-BLEND3", oof_blend3, pub_y)

oof_z_top3 = (
    _rank01(pub_X[:, Z_PHI])
    + _rank01(pub_X[:, Z_PHI_MAX])
    + _rank01(pub_X[:, Z_LOSS_MIN])
) / 3.0
priv_z_top3 = (
    _rank01(priv_X[:, Z_PHI])
    + _rank01(priv_X[:, Z_PHI_MAX])
    + _rank01(priv_X[:, Z_LOSS_MIN])
) / 3.0
t_z_top3 = _eval("OOF-Z-TOP3-RANK", oof_z_top3, pub_y)

oof_phi_robust_pair = (
    _rank01(pub_X[:, Z_PHI]) + _rank01(pub_X[:, Z_PHI_R])
) / 2.0
priv_phi_robust_pair = (
    _rank01(priv_X[:, Z_PHI]) + _rank01(priv_X[:, Z_PHI_R])
) / 2.0
t_phi_robust_pair = _eval("OOF-PHI-ROBUST-PAIR", oof_phi_robust_pair, pub_y)

oof_phi_core_rank = (
    _rank01(pub_X[:, Z_PHI]) + _rank01(oof_lr_core)
) / 2.0
priv_phi_core_rank = (
    _rank01(priv_X[:, Z_PHI]) + _rank01(priv_lr_core)
) / 2.0
t_phi_core_rank = _eval("OOF-PHI-CORE-RANK", oof_phi_core_rank, pub_y)

oof_phi_online_rank = (
    _rank01(pub_X[:, Z_PHI])
    + _rank01(pub_X[:, ONLINE_LL_PHI])
    + _rank01(pub_X[:, ONLINE_SIMPLE_PHI])
) / 3.0
priv_phi_online_rank = (
    _rank01(priv_X[:, Z_PHI])
    + _rank01(priv_X[:, ONLINE_LL_PHI])
    + _rank01(priv_X[:, ONLINE_SIMPLE_PHI])
) / 3.0
t_phi_online_rank = _eval("OOF-PHI-ONLINE-RANK", oof_phi_online_rank, pub_y)

oof_online_S_dual = (
    _rank01(pub_X[:, ONLINE_SIMPLE_PHI]) + _rank01(pub_X[:, ONLINE_SIMPLE_LOSS])
) / 2.0
priv_online_S_dual = (
    _rank01(priv_X[:, ONLINE_SIMPLE_PHI]) + _rank01(priv_X[:, ONLINE_SIMPLE_LOSS])
) / 2.0
t_online_S_dual = _eval("OOF-ONLINE-S-DUAL", oof_online_S_dual, pub_y)

oof_mlp_Sphi = (
    _rank01(oof_mlp) + _rank01(pub_X[:, ONLINE_SIMPLE_PHI])
) / 2.0
priv_mlp_Sphi = (
    _rank01(priv_mlp) + _rank01(priv_X[:, ONLINE_SIMPLE_PHI])
) / 2.0
t_mlp_Sphi = _eval("OOF-MLP-SPHI", oof_mlp_Sphi, pub_y)

oof_mia_mega = (
    _rank01(pub_X[:, ONLINE_SIMPLE_PHI])
    + _rank01(pub_X[:, ONLINE_SIMPLE_LOSS])
    + _rank01(oof_mlp)
) / 3.0
priv_mia_mega = (
    _rank01(priv_X[:, ONLINE_SIMPLE_PHI])
    + _rank01(priv_X[:, ONLINE_SIMPLE_LOSS])
    + _rank01(priv_mlp)
) / 3.0
t_mia_mega = _eval("OOF-MIA-MEGA", oof_mia_mega, pub_y)

oof_mia_quad = (
    _rank01(pub_X[:, ONLINE_SIMPLE_PHI])
    + _rank01(pub_X[:, ONLINE_SIMPLE_LOSS])
    + _rank01(oof_mlp)
    + _rank01(oof_lr_full)
) / 4.0
priv_mia_quad = (
    _rank01(priv_X[:, ONLINE_SIMPLE_PHI])
    + _rank01(priv_X[:, ONLINE_SIMPLE_LOSS])
    + _rank01(priv_mlp)
    + _rank01(priv_lr_full)
) / 4.0
t_mia_quad = _eval("OOF-MIA-QUAD", oof_mia_quad, pub_y)

oof_mia_penta = (
    _rank01(pub_X[:, ONLINE_SIMPLE_PHI])
    + _rank01(pub_X[:, ONLINE_SIMPLE_LOSS])
    + _rank01(pub_X[:, ONLINE_LL_PHI])
    + _rank01(oof_mlp)
    + _rank01(oof_lr_core)
) / 5.0
priv_mia_penta = (
    _rank01(priv_X[:, ONLINE_SIMPLE_PHI])
    + _rank01(priv_X[:, ONLINE_SIMPLE_LOSS])
    + _rank01(priv_X[:, ONLINE_LL_PHI])
    + _rank01(priv_mlp)
    + _rank01(priv_lr_core)
) / 5.0
t_mia_penta = _eval("OOF-MIA-PENTA", oof_mia_penta, pub_y)


candidates = {
    "lr_full":      (t_lr_full,     oof_lr_full,             priv_lr_full),
    "lr_core":      (t_lr_core,     oof_lr_core,             priv_lr_core),
    "lr_blend":     (t_lr_blend,    oof_lr_blend,            priv_lr_blend),
    "mlp":          (t_mlp,         oof_mlp,                 priv_mlp),
    "lr_perclass":  (t_lr_perclass, oof_lr_perclass,         priv_lr_perclass),
    "blend3":       (t_blend3,      oof_blend3,              priv_blend3),
    "super_blend":  (t_super,       oof_super,               priv_super),
    "z_phi":        (_tpr5(pub_y, pub_X[:, Z_PHI]),         pub_X[:, Z_PHI],         priv_X[:, Z_PHI]),
    "z_phi_max":    (_tpr5(pub_y, pub_X[:, Z_PHI_MAX]),     pub_X[:, Z_PHI_MAX],     priv_X[:, Z_PHI_MAX]),
    "z_loss":       (_tpr5(pub_y, pub_X[:, Z_LOSS]),        pub_X[:, Z_LOSS],        priv_X[:, Z_LOSS]),
    "z_loss_min":   (_tpr5(pub_y, pub_X[:, Z_LOSS_MIN]),    pub_X[:, Z_LOSS_MIN],    priv_X[:, Z_LOSS_MIN]),
    "z_loss_R":     (_tpr5(pub_y, pub_X[:, Z_LOSS_R]),      pub_X[:, Z_LOSS_R],      priv_X[:, Z_LOSS_R]),
    "z_loss_min_R": (_tpr5(pub_y, pub_X[:, Z_LOSS_MIN_R]),  pub_X[:, Z_LOSS_MIN_R],  priv_X[:, Z_LOSS_MIN_R]),
    "z_phi_R":      (_tpr5(pub_y, pub_X[:, Z_PHI_R]),       pub_X[:, Z_PHI_R],       priv_X[:, Z_PHI_R]),
    "z_phi_max_R":  (_tpr5(pub_y, pub_X[:, Z_PHI_MAX_R]),   pub_X[:, Z_PHI_MAX_R],   priv_X[:, Z_PHI_MAX_R]),
    "online_LL_phi": (_tpr5(pub_y, pub_X[:, ONLINE_LL_PHI]),  pub_X[:, ONLINE_LL_PHI],  priv_X[:, ONLINE_LL_PHI]),
    "online_LL_loss": (_tpr5(pub_y, pub_X[:, ONLINE_LL_LOSS]), pub_X[:, ONLINE_LL_LOSS], priv_X[:, ONLINE_LL_LOSS]),
    "online_S_phi": (_tpr5(pub_y, pub_X[:, ONLINE_SIMPLE_PHI]),  pub_X[:, ONLINE_SIMPLE_PHI],  priv_X[:, ONLINE_SIMPLE_PHI]),
    "online_S_loss":(_tpr5(pub_y, pub_X[:, ONLINE_SIMPLE_LOSS]), pub_X[:, ONLINE_SIMPLE_LOSS], priv_X[:, ONLINE_SIMPLE_LOSS]),
    "online_S_dual": (t_online_S_dual, oof_online_S_dual, priv_online_S_dual),
    "mlp_Sphi": (t_mlp_Sphi, oof_mlp_Sphi, priv_mlp_Sphi),
    "mia_mega": (t_mia_mega, oof_mia_mega, priv_mia_mega),
    "mia_quad": (t_mia_quad, oof_mia_quad, priv_mia_quad),
    "mia_penta": (t_mia_penta, oof_mia_penta, priv_mia_penta),
}

oof_rank_triplet = (
    _rank01(pub_X[:, Z_PHI_MAX])
    + _rank01(pub_X[:, Z_LOSS])
    + _rank01(pub_X[:, ONLINE_SIMPLE_LOSS])
) / 3.0
priv_rank_triplet = (
    _rank01(priv_X[:, Z_PHI_MAX])
    + _rank01(priv_X[:, Z_LOSS])
    + _rank01(priv_X[:, ONLINE_SIMPLE_LOSS])
) / 3.0
t_rank_triplet = _tpr5(pub_y, oof_rank_triplet)
candidates["rank_triplet"] = (t_rank_triplet, oof_rank_triplet, priv_rank_triplet)
candidates["z_top3_rank"] = (t_z_top3, oof_z_top3, priv_z_top3)
candidates["phi_robust_pair"] = (t_phi_robust_pair, oof_phi_robust_pair, priv_phi_robust_pair)
candidates["phi_core_rank"] = (t_phi_core_rank, oof_phi_core_rank, priv_phi_core_rank)
candidates["phi_online_rank"] = (t_phi_online_rank, oof_phi_online_rank, priv_phi_online_rank)

print("=" * 60)
print("Candidate summary (pub OOF TPR@5%FPR):")
for k, (t, _, _) in sorted(candidates.items(), key=lambda kv: -kv[1][0]):
    print(f"  {k:>16s}: {t:.4f}")

plain_best = max(candidates, key=lambda k: candidates[k][0])


LEARNED = {"lr_full", "lr_core", "lr_blend", "mlp", "lr_perclass",
           "blend3", "super_blend"}
ENSEMBLE_LEARNED = {
    "blend3",
    "super_blend",
    "rank_triplet",
    "z_top3_rank",
    "phi_robust_pair",
    "phi_core_rank",
    "phi_online_rank",
    "online_S_dual",
    "mlp_Sphi",
    "mia_mega",
    "mia_quad",
    "mia_penta",
}
non_learned_best = max((k for k in candidates if k not in LEARNED),
                       key=lambda k: candidates[k][0])
OCCAM_MARGIN = 0.003
marginal = (
    candidates[plain_best][0] - candidates[non_learned_best][0]
)
if (
    plain_best in LEARNED
    and plain_best not in ENSEMBLE_LEARNED
    and 0 < marginal < OCCAM_MARGIN
):
    best_name = non_learned_best
    print(f"Occam veto: {plain_best} ({candidates[plain_best][0]:.4f}) "
          f"only beats {non_learned_best} ({candidates[non_learned_best][0]:.4f}) "
          f"by < {OCCAM_MARGIN:.3f} -> use {non_learned_best}")
else:
    best_name = plain_best


ROBUST_SUBMIT_POOL = frozenset(ENSEMBLE_LEARNED)
PRIV_PUB_GAP_GUARD = 0.0055
champion_t = candidates[plain_best][0]
if plain_best not in ROBUST_SUBMIT_POOL:
    best_robust = max(ROBUST_SUBMIT_POOL, key=lambda k: candidates[k][0])
    t_rob = candidates[best_robust][0]
    if champion_t - t_rob <= PRIV_PUB_GAP_GUARD:
        if best_name != best_robust:
            print(
                f"\nPriv/OOF guard: pub champion {plain_best} ({champion_t:.4f}) vs "
                f"best robust fuse {best_robust} ({t_rob:.4f}), gap {champion_t - t_rob:.4f} "
                f"<= {PRIV_PUB_GAP_GUARD:.4f} -> submit {best_robust} "
                f"(was {best_name})."
            )
        best_name = best_robust

best_t, best_oof, best_priv = candidates[best_name]
print(f"\nSELECTED: {best_name} -> OOF TPR@5%FPR = {best_t:.4f}")


ids = [target_combined[N_PUB + i]["id"] for i in range(N_PRIV)]
scores = _rank01(best_priv)  # rank-normalise to [0, 1]

assignment_tpr5_ranked = _tpr5(pub_y, _rank01(best_oof))
print()
print("=" * 72)
print("ASSIGNMENT METRIC  |  Score = TPR@5%FPR  (Trustworthy ML 2026, MIA task)")
print("=" * 72)
print(f"  Selected attack:           {best_name}")
print(f"  Score (TPR@5%FPR), pub OOF, rank-normalized like submission.csv:")
print(f"                             {assignment_tpr5_ranked:.6f}")
print(f"  Same metric on raw OOF scores (model selection used this):")
print(f"                             {best_t:.6f}")
print("  (Leaderboard = identical metric on hidden priv.pt membership.)")
print("=" * 72)

df = pd.DataFrame({"id": ids, "score": scores})
df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)


# submit
def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)

parser = argparse.ArgumentParser(description="Submit a CSV file to the server.")
args = parser.parse_args()

submit_path = OUTPUT_CSV

if not submit_path.exists():
    die(f"File not found: {submit_path}")

try:
    with open(submit_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": API_KEY},
            files={"file": (submit_path.name, f, "application/csv")},
            timeout=(10, 600),
        )
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}

    if resp.status_code == 413:
        die("Upload rejected: file too large (HTTP 413).")

    resp.raise_for_status()

    print("Successfully submitted.")
    print("Server response:", body)
    submission_id = body.get("submission_id")
    if submission_id:
        print(f"Submission ID: {submission_id}")

except requests.exceptions.RequestException as e:
    detail = getattr(e, "response", None)
    print(f"Submission error: {e}")
    if detail is not None:
        try:
            print("Server response:", detail.json())
        except Exception:
            print("Server response (text):", detail.text)
    sys.exit(1)
