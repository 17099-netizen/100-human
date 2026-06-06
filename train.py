from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from dataset_loader import MultiImageFolderDataset
from model import MultiImageLivenessNet


def build_parser():
    p = argparse.ArgumentParser(description="Train Human3D-AI baseline model")
    p.add_argument("--dataset", type=str, default="dataset", help="Dataset root folder")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--max-images", type=int, default=5)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrained", action="store_true", help="Use torchvision pretrained backbone")
    p.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone weights")
    p.add_argument("--save-path", type=str, default="model/best.pt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main():
    args = build_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    dataset = MultiImageFolderDataset(
        args.dataset,
        image_size=args.image_size,
        max_images=args.max_images,
        transform=transform,
    )

    val_size = max(1, int(len(dataset) * args.val_split)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size

    if val_size == 0:
        train_ds = dataset
        val_ds = None
    else:
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0) if val_ds else None

    device = torch.device(args.device)
    model = MultiImageLivenessNet(pretrained=args.pretrained, freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        val_acc = None
        if val_loader is not None:
            model.eval()
            vcorrect = 0
            vtotal = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    logits = model(images)
                    preds = logits.argmax(dim=1)
                    vcorrect += (preds == labels).sum().item()
                    vtotal += labels.size(0)
            val_acc = vcorrect / max(1, vtotal)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "image_size": args.image_size,
                        "max_images": args.max_images,
                        "class_names": ["fake", "human"],
                        "pretrained": args.pretrained,
                    },
                    save_path,
                )
        else:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": args.image_size,
                    "max_images": args.max_images,
                    "class_names": ["fake", "human"],
                    "pretrained": args.pretrained,
                },
                save_path,
            )

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "train_acc": round(train_acc, 6),
                    "val_acc": None if val_acc is None else round(val_acc, 6),
                    "saved_to": str(save_path),
                },
                ensure_ascii=False,
            )
        )

    print(f"Done. Best model: {save_path}")


if __name__ == "__main__":
    main()
