"""
Garbage Classification 12 คลาส ด้วย EfficientNet-V2-S (transfer learning จาก ImageNet)

เทรน 2 ขั้น:
  Stage 1 : freeze backbone เทรนเฉพาะหัว classifier
  Stage 2 : ปลด freeze เทรนทั้งโมเดล (backbone lr ต่ำ, head lr สูงกว่า, OneCycleLR)

โครงข้อมูล: data/{train,val,test}/{class}/*.jpg

รัน baseline : python train_efficientnet.py
ablation     : python train_efficientnet.py --tag no_pretrained --no_pretrained
               python train_efficientnet.py --tag no_aug --no_aug
               python train_efficientnet.py --tag no_2stage --head_epochs 0
               python train_efficientnet.py --tag no_smooth --label_smoothing 0
               python train_efficientnet.py --tag no_cw --no_class_weight
               python train_efficientnet.py --tag high_lr --backbone_lr 1e-3
ผลทั้งหมดอยู่ที่ saves/<tag>/ (checkpoint, history.csv, กราฟ, results.json)
"""

import os, json, copy, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support


# ------------------------------------------------------------ config
def get_config():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="saves")
    p.add_argument("--tag", default="baseline", help="ชื่อการทดลอง (ใช้ตั้งชื่อโฟลเดอร์ผลลัพธ์)")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--head_epochs", type=int, default=3)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--ft_epochs", type=int, default=15)
    p.add_argument("--backbone_lr", type=float, default=1e-4)
    p.add_argument("--head_lr_ft", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--no_class_weight", action="store_true")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--no_aug", action="store_true")
    return p.parse_args()


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ------------------------------------------------------------ data
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_loaders(cfg):
    norm = transforms.Normalize(MEAN, STD)
    eval_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(), norm])
    if cfg.no_aug:
        train_tf = eval_tf
    else:
        # ไม่ปรับ hue เพราะขวดแก้วต่างสีต่างคลาส
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(cfg.img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(), norm])

    train_set = datasets.ImageFolder(os.path.join(cfg.data, "train"), transform=train_tf)
    val_set = datasets.ImageFolder(os.path.join(cfg.data, "val"), transform=eval_tf)
    test_set = datasets.ImageFolder(os.path.join(cfg.data, "test"), transform=eval_tf)
    assert train_set.classes == val_set.classes == test_set.classes

    mk = lambda ds, sh: DataLoader(ds, batch_size=cfg.batch, shuffle=sh, num_workers=cfg.workers)
    return mk(train_set, True), mk(val_set, False), mk(test_set, False), train_set, test_set


# ------------------------------------------------------------ model
def build_model(cfg, num_classes):
    weights = None if cfg.no_pretrained else models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = models.efficientnet_v2_s(weights=weights)
    in_f = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_f, num_classes))
    return model


# ------------------------------------------------------------ train / eval
def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    train = optimizer is not None
    model.train(train)
    loss_sum = correct = total = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), 1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


# ------------------------------------------------------------ plots
def plot_curves(hist, head_epochs, best_ep, path):
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for a, k, t in [(ax[0], "acc", "Accuracy"), (ax[1], "loss", "Loss")]:
        a.plot(ep, [h["train_" + k] for h in hist], "o-", label="train")
        a.plot(ep, [h["val_" + k] for h in hist], "o-", label="val")
        a.axvline(head_epochs + 0.5, color="gray", ls="--")
        a.axvline(best_ep, color="green", ls=":")
        a.set_title(t); a.set_xlabel("epoch"); a.grid(alpha=.3); a.legend()
    fig.suptitle("EfficientNet-V2-S (gray = unfreeze, green = best epoch)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_confusion(y, pred, classes, path):
    cm = confusion_matrix(y, pred)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    for a, m, t, f in [(ax[0], cm, "counts", "d"), (ax[1], cmn, "normalized", ".2f")]:
        a.imshow(m, cmap="Blues")
        a.set_xticks(range(len(classes))); a.set_xticklabels(classes, rotation=45, ha="right")
        a.set_yticks(range(len(classes))); a.set_yticklabels(classes)
        for i in range(len(classes)):
            for j in range(len(classes)):
                a.text(j, i, format(m[i, j], f), ha="center", va="center",
                       color="white" if m[i, j] > m.max() / 2 else "black", fontsize=7)
        a.set_title(f"Confusion Matrix ({t})"); a.set_xlabel("Predicted"); a.set_ylabel("True")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_per_class(y, pred, classes, path):
    p, r, f, _ = precision_recall_fscore_support(y, pred)
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for i, (v, n) in enumerate([(p, "precision"), (r, "recall"), (f, "f1")]):
        ax.bar(x + (i - 1) * 0.27, v, 0.27, label=n)
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylim(max(0, min(p.min(), r.min()) - 0.05), 1.0)
    ax.set_title("Per-class Precision / Recall / F1 (test)"); ax.grid(axis="y", alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_top_errors(probs, y, pred, test_set, classes, path, n=12):
    wrong = np.where(pred != y)[0]
    wrong = wrong[np.argsort(-probs[wrong].max(1))][:n]
    if len(wrong) == 0:
        return
    cols = 4
    rows = (len(wrong) + cols - 1) // cols
    fig, ax = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.3 * rows))
    ax = np.atleast_2d(ax)
    for a in ax.ravel():
        a.axis("off")
    for a, i in zip(ax.ravel(), wrong):
        a.imshow(Image.open(test_set.samples[i][0]).convert("RGB"))
        a.set_title(f"true: {classes[y[i]]}\npred: {classes[pred[i]]} ({probs[i].max():.2f})",
                    color="red", fontsize=9)
    fig.suptitle(f"Most confident mistakes (total wrong = {(pred != y).sum()} / {len(y)})")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ------------------------------------------------------------ main
def main():
    cfg = get_config()
    torch.manual_seed(cfg.seed)
    device = get_device()
    out = os.path.join(cfg.out, cfg.tag)
    os.makedirs(out, exist_ok=True)

    train_loader, val_loader, test_loader, train_set, test_set = get_loaders(cfg)
    classes = train_set.classes
    nc = len(classes)
    print(f"Device: {device} | tag: {cfg.tag} | classes: {nc}")
    print(f"train/val/test = {len(train_set)}/{len(val_loader.dataset)}/{len(test_set)}")

    model = build_model(cfg, nc).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.2f} M")
    with open(os.path.join(out, "model_structure.txt"), "w") as f:
        f.write(str(model))

    # class weight = N / (nc * n_c)  (คลาสที่รูปน้อยได้น้ำหนักมาก)
    counts = np.bincount(train_set.targets, minlength=nc)
    cw = None if cfg.no_class_weight else torch.tensor(
        len(train_set) / (nc * counts), dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=cfg.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()  # val loss แบบไม่ smooth/ไม่ถ่วง เทียบกันได้ทุกการทดลอง

    hist, best_acc, best_w, best_ep = [], 0.0, None, 0
    t0 = time.time()

    def log(ep, stage, tl, ta, lr):
        nonlocal best_acc, best_w, best_ep
        vl, va = run_epoch(model, val_loader, eval_criterion, device)
        hist.append(dict(epoch=ep, stage=stage, train_loss=tl, train_acc=ta,
                         val_loss=vl, val_acc=va, lr=lr))
        print(f"[{stage:4s}] ep {ep:2d} | loss {tl:.4f} | train {ta:.4f} | val {va:.4f} | lr {lr:.1e}")
        if va > best_acc:
            best_acc, best_w, best_ep = va, copy.deepcopy(model.state_dict()), ep

    # ---- Stage 1: head only
    if cfg.head_epochs > 0:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
        opt = torch.optim.AdamW(model.classifier.parameters(), lr=cfg.head_lr,
                                weight_decay=cfg.weight_decay)
        for ep in range(1, cfg.head_epochs + 1):
            tl, ta = run_epoch(model, train_loader, criterion, device, opt)
            log(ep, "head", tl, ta, cfg.head_lr)

    # ---- Stage 2: fine-tune all
    for p in model.parameters():
        p.requires_grad = True
    backbone = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    opt = torch.optim.AdamW([{"params": backbone, "lr": cfg.backbone_lr},
                             {"params": model.classifier.parameters(), "lr": cfg.head_lr_ft}],
                            weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[cfg.backbone_lr, cfg.head_lr_ft],
        total_steps=cfg.ft_epochs * len(train_loader), pct_start=0.1)
    for ep in range(1, cfg.ft_epochs + 1):
        lr = opt.param_groups[0]["lr"]
        tl, ta = run_epoch(model, train_loader, criterion, device, opt, sch)
        log(cfg.head_epochs + ep, "full", tl, ta, lr)

    train_min = (time.time() - t0) / 60
    torch.save({"state_dict": best_w, "classes": classes, "best_epoch": best_ep,
                "best_val_acc": best_acc}, os.path.join(out, "best.pth"))

    # ---- test ด้วย checkpoint ที่ val ดีที่สุด
    model.load_state_dict(best_w)
    probs, y = predict(model, test_loader, device)
    pred = probs.argmax(1)
    test_acc = float((pred == y).mean())
    top3 = float(np.mean([y[i] in np.argsort(-probs[i])[:3] for i in range(len(y))]))
    p, r, f, _ = precision_recall_fscore_support(y, pred)
    print(f"\nBest epoch {best_ep} | val {best_acc:.4f} | TEST acc {test_acc:.4f} | "
          f"macro F1 {f.mean():.4f} | top3 {top3:.4f}")
    print(classification_report(y, pred, target_names=classes, digits=4))

    # ---- บันทึกผล
    import csv
    with open(os.path.join(out, "history.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hist[0].keys()); w.writeheader(); w.writerows(hist)
    json.dump(dict(tag=cfg.tag, params_M=n_params / 1e6, best_epoch=best_ep, val_acc=best_acc,
                   test_acc=test_acc, macro_f1=float(f.mean()), top3=top3,
                   wrong=int((pred != y).sum()), n_test=len(y), train_minutes=train_min,
                   per_class_f1=dict(zip(classes, map(float, f))), config=vars(cfg)),
              open(os.path.join(out, "results.json"), "w"), indent=2, ensure_ascii=False)

    plot_curves(hist, cfg.head_epochs, best_ep, os.path.join(out, "training-curves.png"))
    plot_confusion(y, pred, classes, os.path.join(out, "confusion-matrix.png"))
    plot_per_class(y, pred, classes, os.path.join(out, "per-class-metrics.png"))
    plot_top_errors(probs, y, pred, test_set, classes, os.path.join(out, "top-errors.png"))
    print("saved ->", out)


if __name__ == "__main__":
    main()