
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


# Config

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

if os.path.exists("/kaggle/working"):
    WORKING_DIR = "/kaggle/working"
else:
    WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

CKPT_DIR = os.path.join(WORKING_DIR, "checkpoints")
PRED_DIR = os.path.join(WORKING_DIR, "predictions")
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model Architecture

class ResBlock(nn.Module):
    def __init__(self, in_channels, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, in_channels)

    def forward(self, input):
        residual = input
        out = F.relu(self.gn1(self.conv1(input)))
        out = self.gn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.gn = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        return F.relu(self.gn(self.conv(x)))


class RobustClassifier(nn.Module):
    """
    RobustClassifier for Hackenza 2026.
    Implements the required interface:
      - __init__(): builds the architecture
      - forward(x): x shape [B, 1, 28, 28] -> logits [B, 10]
      - load_weights(path): loads saved weights
    """
    def __init__(self):
        super().__init__()

        # Stem: 1 channel -> 64 channels, preserve 28x28
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )

        # Residual 1
        self.res1 = ResBlock(64, num_groups=8)

        # Transition 1: 64->128 channels, 28->14 spatial
        self.trans1 = TransitionBlock(64, 128, num_groups=8)

        # Residual 2
        self.res2 = ResBlock(128, num_groups=8)

        # Transition 2: 128->256 channels, 14->7 spatial
        self.trans2 = TransitionBlock(128, 256, num_groups=16)

        # Residual 3
        self.res3 = ResBlock(256, num_groups=16)

        # Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        # x shape: [B, 1, 28, 28]
        x = self.stem(x)
        x = self.res1(x)
        x = self.trans1(x)
        x = self.res2(x)
        x = self.trans2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # logits [B, 10]

    def load_weights(self, path):
        """Load saved model weights."""
        state = torch.load(path, map_location='cpu')
        # Handle both formats: raw state_dict or checkpoint dict
        if isinstance(state, dict) and "model_state_dict" in state:
            self.load_state_dict(state["model_state_dict"])
        else:
            self.load_state_dict(state)
        print(f"Weights loaded from: {path}")

    def get_features(self, x):
        """Returns features before the FC layer."""
        x = self.stem(x)
        x = self.res1(x)
        x = self.trans1(x)
        x = self.res2(x)
        x = self.trans2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x


# Data Loading Utilities

class LabeledDataset(Dataset):
    def __init__(self, path, transform=None):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required dataset file not found: {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.images = data["images"]
        self.labels = data["labels"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class UnlabeledDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img


def get_val_loader(batch_size=100, shuffle=False):
    path = os.path.join(DATA_DIR, "val_sanity.pt")
    ds = LabeledDataset(path, transform=None)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=(DEVICE.type == "cuda"))


def get_target_loader(images, batch_size=256, shuffle=False):
    ds = UnlabeledDataset(images, transform=None)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=(DEVICE.type == "cuda"))


def load_static():
    path = os.path.join(DATA_DIR, "static.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"static.pt not found at {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["images"]


def load_test_suite():
    path = os.path.join(DATA_DIR, "test_suite_public.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"test_suite_public.pt not found at {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data


def ensure_dirs():
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)


# Phase 2: BBSE-Soft Label Shift Estimation

@torch.no_grad()
def build_confusion_matrix(model, val_loader, device, num_classes=10):
    """Build soft confusion matrix C from validation set."""
    model.eval()
    C = torch.zeros(num_classes, num_classes, device=device)
    class_counts = torch.zeros(num_classes, device=device)

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)

        for i in range(images.size(0)):
            y = labels[i].item()
            C[:, y] += probs[i]
            class_counts[y] += 1

    for y in range(num_classes):
        if class_counts[y] > 0:
            C[:, y] /= class_counts[y]

    return C


@torch.no_grad()
def compute_mean_prediction(model, target_loader, device):
    """Compute mean softmax prediction on target data."""
    model.eval()
    mu = torch.zeros(NUM_CLASSES, device=device)
    total = 0

    for batch in target_loader:
        images = batch if not isinstance(batch, (list, tuple)) else batch[0]
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        mu += probs.sum(dim=0)
        total += images.size(0)

    if total == 0:
        raise RuntimeError("Target loader yielded zero samples.")
    mu /= float(total)
    return mu


def estimate_weights(C, mu, reg=1e-4):
    """Estimate label shift weights via BBSE with Tikhonov regularization."""
    C_cpu = C.cpu().double()
    mu_cpu = mu.cpu().double()

    K = C_cpu.shape[0]
    I = torch.eye(K, dtype=C_cpu.dtype, device=C_cpu.device)
    C_reg = C_cpu.T @ C_cpu + reg * I
    rhs = C_cpu.T @ mu_cpu

    try:
        w = torch.linalg.solve(C_reg, rhs)
    except torch.linalg.LinAlgError:
        C_pinv = torch.linalg.pinv(C_reg)
        w = C_pinv @ rhs

    w = w.to(torch.float32)

    def project_simplex(v):
        v_np = v.detach().cpu().numpy()
        u = -np.sort(-v_np)
        cssv = np.cumsum(u) - 1
        rho_arr = u - cssv / (np.arange(1, len(u) + 1))
        rho_candidates = np.where(rho_arr > 0)[0]
        if rho_candidates.size == 0:
            return torch.full_like(v, 1.0 / v.numel())
        rho = rho_candidates[-1]
        theta = cssv[rho] / (rho + 1.0)
        w_proj = torch.from_numpy(np.maximum(v_np - theta, 0.0)).to(v.device).to(v.dtype)
        return w_proj

    w = project_simplex(w)
    return w


def run_estimation(model, device, reg=1e-4):
    """Phase 2: Estimate label shift weights for static and all scenarios."""
    print("\n Phase 2: Label Shift Estimation (BBSE-Soft)")
    print("+" * 60)

    val_loader = get_val_loader()
    C = build_confusion_matrix(model, val_loader, device, NUM_CLASSES)
    print(f"Confusion matrix built from val_sanity.pt")

    weights_dict = {}

    # Static set
    print(f"\n +++ static.pt +++")
    static_images = load_static()
    static_loader = get_target_loader(static_images, batch_size=256)
    mu_static = compute_mean_prediction(model, static_loader, device)
    w_static = estimate_weights(C, mu_static, reg=reg)
    weights_dict["static"] = w_static
    print(f"  Estimated weights: {w_static.numpy().round(4)}")

    # Test scenarios
    test_suite = load_test_suite()
    scenario_keys = [k for k, v in test_suite.items()
                     if isinstance(k, str) and k.startswith("scenario_") and torch.is_tensor(v)]

    for scenario_name in sorted(scenario_keys):
        images = test_suite[scenario_name]
        loader = get_target_loader(images, batch_size=256)
        mu = compute_mean_prediction(model, loader, device)
        w = estimate_weights(C, mu, reg=reg)
        weights_dict[scenario_name] = w
        print(f"  {scenario_name}: n={len(images)}, "
              f"w_max={w.max():.3f} (class {w.argmax().item()})")

    print("+" * 60)
    return weights_dict


# Phase 3: SAR Test-Time Adaptation

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer."""
    def __init__(self, params, base_optimizer_cls, rho=0.05, **kwargs):
        defaults = dict(rho=rho)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(params, **kwargs)

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale
                p.add_(e_w)

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()

    def _grad_norm(self):
        norms = []
        shared_device = self.param_groups[0]["params"][0].device
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    norms.append(p.grad.norm(p=2).to(shared_device))
        if len(norms) == 0:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)


def entropy(probs):
    """Compute per-sample entropy: H(p) = -sum(p * log(p))."""
    return -(probs * torch.log(probs.clamp(min=1e-7))).sum(dim=1)


def prepare_model_for_tta(model):
    """Freeze all parameters except GroupNorm affine (gamma, beta) and FC."""
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    for module in model.modules():
        if isinstance(module, nn.GroupNorm):
            module.weight.requires_grad_(True)
            module.bias.requires_grad_(True)

    if hasattr(model, 'fc'):
        model.fc.weight.requires_grad_(True)
        model.fc.bias.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  TTA params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    return model


def adapt_and_predict(model, target_images, weights, device,
                      num_steps=1, batch_size=256, lr=1e-3, rho=0.05,
                      entropy_threshold=0.4, teacher_momentum=None, distill_weight=0.0):
    """
    SAR: Adapt model on target data and predict.
    Uses entropy-based reliable sample filtering and SAM optimizer.
    """
    adapted_model = copy.deepcopy(model)
    adapted_model = prepare_model_for_tta(adapted_model)

    trainable_params = [p for p in adapted_model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable params for TTA.")

    sam_optimizer = SAM(trainable_params, torch.optim.SGD, rho=rho, lr=lr, momentum=0.9)

    # Optional teacher (CoTTA-style)
    teacher_model = None
    if teacher_momentum is not None and teacher_momentum > 0.0:
        teacher_model = copy.deepcopy(adapted_model)
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        teacher_model.eval()

    log_w = torch.log(weights.to(device).clamp(min=1e-7))
    target_loader = get_target_loader(target_images, batch_size=batch_size, shuffle=False)

    all_predictions = []
    reliable_count = 0
    total_count = 0

    ent_mu_ema = None
    ent_std_ema = None
    ema_alpha = 0.1

    for images in target_loader:
        if isinstance(images, (list, tuple)):
            images = images[0]
        images = images.to(device)

        # Compute entropy and reliable mask
        adapted_model.eval()
        with torch.no_grad():
            logits = adapted_model(images)
            corrected_logits = logits + log_w.unsqueeze(0)
            probs = F.softmax(corrected_logits, dim=1)
            H = entropy(probs)
            batch_mu = float(H.mean().item())
            batch_std = float(H.std().item()) if H.numel() > 1 else 0.0

            if ent_mu_ema is None:
                ent_mu_ema = batch_mu
                ent_std_ema = batch_std
            else:
                ent_mu_ema = (1.0 - ema_alpha) * ent_mu_ema + ema_alpha * batch_mu
                ent_std_ema = (1.0 - ema_alpha) * ent_std_ema + ema_alpha * batch_std

            multiplier = 1.0 if entropy_threshold < 0.5 else entropy_threshold
            threshold = ent_mu_ema - multiplier * ent_std_ema
            reliable_mask = (H < threshold)

            if reliable_mask.sum().item() == 0:
                reliable_mask = (H < ent_mu_ema)

        reliable_count += int(reliable_mask.sum().item())
        total_count += images.size(0)

        # Skip adaptation if nothing reliable
        if reliable_mask.sum().item() == 0:
            adapted_model.eval()
            with torch.no_grad():
                logits = adapted_model(images)
                corrected_logits = logits + log_w.unsqueeze(0)
                preds = corrected_logits.argmax(dim=1)
                all_predictions.append(preds.cpu())
            continue

        # SAM adaptation steps
        for _ in range(num_steps):
            adapted_model.train()

            logits = adapted_model(images)
            corrected_logits = logits + log_w.unsqueeze(0)

            reliable_probs = F.softmax(corrected_logits[reliable_mask], dim=1)
            ent_loss = entropy(reliable_probs).mean()

            if teacher_model is not None and distill_weight > 0.0:
                with torch.no_grad():
                    t_logits = teacher_model(images)
                    t_corrected = t_logits + log_w.unsqueeze(0)
                    t_probs = F.softmax(t_corrected, dim=1)
                student_logp = F.log_softmax(corrected_logits[reliable_mask], dim=1)
                teacher_p = t_probs[reliable_mask]
                kl = F.kl_div(student_logp, teacher_p.clamp(min=1e-8), reduction='batchmean')
                total_loss = ent_loss + distill_weight * kl
            else:
                total_loss = ent_loss

            # SAM ascent step
            sam_optimizer.zero_grad()
            total_loss.backward()
            sam_optimizer.first_step()

            # Recompute at perturbed weights -> descent step
            logits_p = adapted_model(images)
            corrected_logits_p = logits_p + log_w.unsqueeze(0)
            reliable_probs_p = F.softmax(corrected_logits_p[reliable_mask], dim=1)
            ent_loss_p = entropy(reliable_probs_p).mean()

            if teacher_model is not None and distill_weight > 0.0:
                with torch.no_grad():
                    t_logits = teacher_model(images)
                    t_corrected = t_logits + log_w.unsqueeze(0)
                    t_probs = F.softmax(t_corrected, dim=1)
                student_logp_p = F.log_softmax(corrected_logits_p[reliable_mask], dim=1)
                teacher_p = t_probs[reliable_mask]
                kl_p = F.kl_div(student_logp_p, teacher_p.clamp(min=1e-8), reduction='batchmean')
                total_loss_p = ent_loss_p + distill_weight * kl_p
            else:
                total_loss_p = ent_loss_p

            sam_optimizer.zero_grad()
            total_loss_p.backward()
            sam_optimizer.second_step()

            # Optional teacher EMA update
            if teacher_model is not None and teacher_momentum is not None:
                with torch.no_grad():
                    for t_p, s_p in zip(teacher_model.parameters(), adapted_model.parameters()):
                        t_p.data.mul_(teacher_momentum).add_(s_p.data * (1.0 - teacher_momentum))

        # Final inference for this batch
        adapted_model.eval()
        with torch.no_grad():
            logits = adapted_model(images)
            corrected_logits = logits + log_w.unsqueeze(0)
            preds = corrected_logits.argmax(dim=1)
            all_predictions.append(preds.cpu())

    predictions = torch.cat(all_predictions, dim=0)
    reliability = 100.0 * reliable_count / max(total_count, 1)
    print(f"    Reliable samples: {reliable_count}/{total_count} ({reliability:.1f}%)")
    return predictions


def run_adaptation(model, device, weights_dict):
    """Phase 3: Run SAR adaptation on static and all scenario sets."""
    print(f"\n Phase 3: Test-Time Adaptation (SAR)")
    print("=" * 60)

    predictions_dict = {}

    # Static set
    print(f"\n  --- static.pt ---")
    static_images = load_static()
    w_static = weights_dict["static"]
    preds_static = adapt_and_predict(
        model, static_images, w_static, device,
        num_steps=3, batch_size=256
    )
    predictions_dict["static"] = preds_static

    # Test scenarios
    test_suite = load_test_suite()
    scenario_keys = [k for k, v in test_suite.items()
                     if isinstance(k, str) and k.startswith("scenario_") and torch.is_tensor(v)]

    for scenario_name in sorted(scenario_keys):
        print(f"\n  +++ {scenario_name} +++")
        images = test_suite[scenario_name]
        w = weights_dict[scenario_name]
        preds = adapt_and_predict(
            model, images, w, device,
            num_steps=3, batch_size=256
        )
        predictions_dict[scenario_name] = preds

    print("=" * 60)
    return predictions_dict


# Submission CSV Generation

def generate_submission(model, static_path, suite_path):
    """
    Generate submission.csv complying with the required format:
      - Columns: ID, Category
      - IDs: static_i for static set, scenario_XX_i for stress test suite
    """
    model.eval()
    device = next(model.parameters()).device
    results = []

    # 1. Evaluate Static Set (Public LB)
    static = torch.load(static_path, map_location="cpu", weights_only=False)
    static_images = static['images'].to(device)
    with torch.no_grad():
        preds = model(static_images).argmax(1)
        for i, p in enumerate(preds):
            results.append({'ID': f'static_{i}', 'Category': int(p)})

    # 2. Evaluate 24-Scenario Suite (Private LB)
    suite = torch.load(suite_path, map_location="cpu", weights_only=False)
    scenario_keys = sorted([k for k in suite.keys() if k.startswith('scenario')])

    for skey in scenario_keys:
        scenario_images = suite[skey].to(device)
        with torch.no_grad():
            preds = model(scenario_images).argmax(1)
            for i, p in enumerate(preds):
                results.append({'ID': f'{skey}_{i}', 'Category': int(p)})

    out_path = os.path.join(WORKING_DIR, 'submission.csv')
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Submission CSV saved: {out_path} ({len(results)} predictions)")
    return out_path


def generate_submission_with_adaptation(model, device, weights_dict, predictions_dict):
    """
    Generate submission.csv from adapted predictions (Phase 2+3).
    Uses the same ID/Category format.
    """
    results = []

    # Static predictions
    if "static" in predictions_dict:
        preds = predictions_dict["static"]
        for i, p in enumerate(preds):
            results.append({'ID': f'static_{i}', 'Category': int(p)})

    # Scenario predictions
    scenario_keys = sorted([k for k in predictions_dict.keys() if k.startswith("scenario_")])
    for skey in scenario_keys:
        preds = predictions_dict[skey]
        for i, p in enumerate(preds):
            results.append({'ID': f'{skey}_{i}', 'Category': int(p)})

    out_path = os.path.join(WORKING_DIR, 'submission.csv')
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Submission CSV saved: {out_path} ({len(results)} predictions)")
    return out_path


# Main: Full Inference Pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hackenza Inference Pipeline")
    parser.add_argument("--weights", type=str, default="weights.pth",
                        help="Path to model weights (default: weights.pth)")
    parser.add_argument("--no-adapt", action="store_true",
                        help="Skip Phase 2+3 adaptation, use direct inference only")
    parser.add_argument("--reg", type=float, default=1e-4,
                        help="Tikhonov regularization for BBSE (default: 1e-4)")
    args = parser.parse_args()

    ensure_dirs()
    device = DEVICE
    print(f"Device: {device}")

    # Load model
    model = RobustClassifier().to(device)
    model.load_weights(args.weights)

    if args.no_adapt:
        # Direct inference without adaptation
        static_path = os.path.join(DATA_DIR, "static.pt")
        suite_path = os.path.join(DATA_DIR, "test_suite_public.pt")
        generate_submission(model, static_path, suite_path)
    else:
        # Full pipeline: Phase 2 (BBSE) + Phase 3 (SAR) + CSV
        weights_dict = run_estimation(model, device, reg=args.reg)
        predictions_dict = run_adaptation(model, device, weights_dict)
        generate_submission_with_adaptation(model, device, weights_dict, predictions_dict)

    print("\nDone!")
