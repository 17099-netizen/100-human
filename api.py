from __future__ import annotations

import argparse
import io
from pathlib import Path

import torch
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import transforms

from model import MultiImageLivenessNet


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def create_app(model_path: str):
    app = Flask(__name__)

    ckpt = torch.load(model_path, map_location="cpu")
    image_size = int(ckpt.get("image_size", 224))
    max_images = int(ckpt.get("max_images", 5))
    pretrained = bool(ckpt.get("pretrained", False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiImageLivenessNet(pretrained=pretrained)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/predict")
    def predict():
        files = request.files.getlist("images")
        if not files:
            return jsonify({"error": "No images uploaded. Use field name 'images'."}), 400

        imgs = []
        for f in files:
            if not f.filename:
                continue
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
            imgs.append(tfm(img))

        if not imgs:
            return jsonify({"error": "No valid images found."}), 400

        if len(imgs) >= max_images:
            imgs = imgs[:max_images]
        else:
            imgs = imgs + [imgs[-1]] * (max_images - len(imgs))

        x = torch.stack(imgs, dim=0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

        return jsonify({
            "fake": float(probs[0]),
            "human": float(probs[1]),
            "predicted": "human" if probs[1] >= probs[0] else "fake",
            "max_images_used": max_images,
        })

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model/best.pt")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    app = create_app(args.model)
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
