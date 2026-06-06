from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import MultiImageLivenessNet


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_images(paths, image_size: int, max_images: int):
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        imgs.append(tfm(img))

    if not imgs:
        raise ValueError("No images provided")

    if len(imgs) >= max_images:
        imgs = imgs[:max_images]
    else:
        imgs = imgs + [imgs[-1]] * (max_images - len(imgs))

    return torch.stack(imgs, dim=0)  # [N,C,H,W]


def gather_paths(args):
    paths = []
    if args.images:
        paths.extend([Path(p) for p in args.images])
    if args.folder:
        folder = Path(args.folder)
        paths.extend(sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()]))
    return [p for p in paths if p.exists()]


def main():
    parser = argparse.ArgumentParser(description="Predict human/fake from multiple images")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--images", nargs="*", default=[])
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    image_size = int(ckpt.get("image_size", 224))
    max_images = int(ckpt.get("max_images", 5))
    pretrained = bool(ckpt.get("pretrained", False))

    device = torch.device(args.device)
    model = MultiImageLivenessNet(pretrained=pretrained)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    paths = gather_paths(args)
    if not paths:
        raise SystemExit("No valid images found. Use --images or --folder.")

    x = load_images(paths, image_size=image_size, max_images=max_images).unsqueeze(0).to(device)  # [1,N,C,H,W]

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

    result = {
        "fake": float(probs[0]),
        "human": float(probs[1]),
        "predicted": "human" if probs[1] >= probs[0] else "fake",
        "num_input_images": len(paths),
        "max_images_used": max_images,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
