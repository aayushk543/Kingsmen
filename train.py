import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import time
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as T


# Reproducibility

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# Config

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

if os.path.exists("/kaggle/working"):
    WORKING_DIR = "/kaggle/working"
else:
    WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

CKPT_DIR = os.path.join(WORKING_DIR, "checkpoints")
NUM_CLASSES = 10
IMG_SIZE = 28
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model Architecture

class ResBlock(nn.Module):
    """Residual block with Group Normalization."""
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
    """Transition block: doubles channels + halves spatial dimension."""
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.gn = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        return F.relu(self.gn(self.conv(x)))


class CustomResNet(nn.Module):
    """Custom ResNet with Group Normalization for noise-robust training."""
    def __init__(self, num_classes=10):
        super().__init__()

        # Stem: 1 channel -> 64 channels, preserve 28x28
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )

        # Residual 1, preserve shape
        self.res1 = ResBlock(64, num_groups=8)

        # Transition 1: 64->128 channels, 28->14 spatial
        self.trans1 = TransitionBlock(64, 128, num_groups=8)

        # Residual 2, preserve shape
        self.res2 = ResBlock(128, num_groups=8)

        # Transition 2: 128->256 channels, 14->7 spatial
        self.trans2 = TransitionBlock(128, 256, num_groups=16)

        # Residual 3, preserve shape
        self.res3 = ResBlock(256, num_groups=16)

        # Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x)
        x = self.trans1(x)
        x = self.res2(x)
        x = self.trans2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# Data Loading

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


def get_train_transform():
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomCrop(IMG_SIZE, padding=4),
        T.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0),
    ])


def get_source_loader(batch_size=128, shuffle=True):
    path = os.path.join(DATA_DIR, "source_toxic.pt")
    ds = LabeledDataset(path, transform=get_train_transform())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=(DEVICE.type == "cuda"))


def get_val_loader(batch_size=100, shuffle=False):
    path = os.path.join(DATA_DIR, "val_sanity.pt")
    ds = LabeledDataset(path, transform=None)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=(DEVICE.type == "cuda"))


def ensure_dirs():
    os.makedirs(CKPT_DIR, exist_ok=True)


def save_checkpoint(model, epoch, val_acc, path):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "val_acc": float(val_acc),
    }
    torch.save(ckpt, path)
    print(f"Checkpoint saved: {path} (val_acc={val_acc:.2f}%)")


# Loss Functions (APL = Active + Passive)

class GCELoss(nn.Module):
    """Generalized Cross Entropy Loss (Active component)."""
    def __init__(self, q=0.7, num_classes=10):
        super().__init__()
        self.q = q
        self.num_classes = num_classes

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        p_y = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_y = p_y.clamp(min=1e-4, max=1.0)
        loss = (1.0 - p_y ** self.q) / self.q
        return loss.mean()


class NormalizedCrossEntropy(nn.Module):
    """Normalized Cross Entropy Loss (Passive component)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        probs = probs.clamp(min=1e-4, max=1.0)
        p_y = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        numerator = -torch.log(p_y)
        denominator = -torch.log(probs).sum(dim=1)
        loss = numerator / denominator.clamp(min=1e-4)
        return loss.mean()


class APLLoss(nn.Module):
    """Active-Passive Loss: APL = alpha * GCE + beta * NCE"""
    def __init__(self, alpha=1.0, beta=1.0, q=0.7, num_classes=10):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gce = GCELoss(q=q, num_classes=num_classes)
        self.nce = NormalizedCrossEntropy(num_classes=num_classes)

    def forward(self, logits, targets):
        loss_gce = self.gce(logits, targets)
        loss_nce = self.nce(logits, targets)
        return self.alpha * loss_gce + self.beta * loss_nce


# Validation

@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    correct, total = 0, 0
    val_loss = 0.0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        val_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    acc = 100.0 * correct / max(1, total)
    avg_val_loss = val_loss / max(1, len(val_loader))
    return acc, avg_val_loss


# Training Loop

def train(args):
    ensure_dirs()
    device = DEVICE
    print(f"Device: {device}")

    # Data
    train_loader = get_source_loader(batch_size=args.batch_size)
    val_loader = get_val_loader()
    print(f"Training: {len(train_loader.dataset)} images")
    print(f"Validation: {len(val_loader.dataset)} images")

    # Model (random init — no pretrained weights)
    model = CustomResNet(num_classes=NUM_CLASSES).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"CustomResNet: {param_count:,} parameters")

    # APL Loss
    criterion = APLLoss(
        alpha=args.alpha, beta=args.beta,
        q=args.q, num_classes=NUM_CLASSES
    )

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )

    # LR Schedule
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training state
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    ckpt_path = os.path.join(CKPT_DIR, "checkpoint_best.pt")

    print(f"\n{'+'*60}")
    print(f"  Phase 1: Robust Training (Decontamination)")
    print(f"  Total Epochs: {args.epochs} | LR: {args.lr} | Batch: {args.batch_size}")
    print(f"{'+'*60}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct, total = 0, 0
        t0 = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        train_acc = 100.0 * correct / max(1, total)
        avg_loss = epoch_loss / len(train_loader)
        val_acc, val_loss = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.1f}% | "
              f"Val Acc: {val_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {lr:.4f} | "
              f"Time: {elapsed:.1f}s")

        # Save best
        if (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(model, epoch, val_acc, ckpt_path)
        else:
            patience_counter += 1

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\n Early stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    print(f"\n{'+'*60}")
    print(f"Finished | Best ValAcc {best_val_acc:.1f}% @ epoch {best_epoch}")
    print(f"{'+'*60}\n")

    # Also save weights.pth (just the state_dict for model_submission.py compatibility)
    weights_path = os.path.join(WORKING_DIR, "weights.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Model weights saved to: {weights_path}")

    return best_val_acc


# CLI

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Robust Training with APL")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Number of training epochs (default: 80)")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Batch size (default: 200)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Initial learning rate (default: 0.1)")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight decay (default: 5e-4)")
    parser.add_argument("--q", type=float, default=0.7,
                        help="GCE truncation parameter (default: 0.7)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Weight for GCE (active) in APL (default: 1.0)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Weight for NCE (passive) in APL (default: 1.0)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience, 0=disabled (default: 15)")
    parser.add_argument("--quick-test", action="store_true",
                        help="Quick test with 5 epochs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.quick_test:
        args.epochs = 5
        args.patience = 0
    train(args)
