from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()])


class MultiImageFolderDataset(Dataset):
    """Dataset layout:

    dataset/
    ├── human/
    │   ├── sample001/
    │   │   ├── 1.jpg
    │   │   └── 2.jpg
    └── fake/
        ├── sample001/
        │   ├── 1.jpg
        │   └── 2.jpg
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int = 224,
        max_images: int = 5,
        transform=None,
    ):
        self.root = Path(root)
        self.image_size = int(image_size)
        self.max_images = int(max_images)
        self.transform = transform or transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

        self.samples: List[Tuple[Path, int]] = []
        for class_name, label in [("human", 1), ("fake", 0)]:
            class_root = self.root / class_name
            if not class_root.exists():
                continue
            for sample_dir in sorted([p for p in class_root.iterdir() if p.is_dir()]):
                if _list_images(sample_dir):
                    self.samples.append((sample_dir, label))

        if not self.samples:
            raise ValueError(
                f"No samples found in {self.root}. "
                "Expected dataset/human/<sample folders> and dataset/fake/<sample folders>."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path):
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx: int):
        sample_dir, label = self.samples[idx]
        image_paths = _list_images(sample_dir)

        if not image_paths:
            raise RuntimeError(f"No images in {sample_dir}")

        # Keep a fixed number of images per sample for easy batching.
        if len(image_paths) >= self.max_images:
            image_paths = image_paths[: self.max_images]
        else:
            # Pad by repeating the last image so every sample has the same shape.
            image_paths = image_paths + [image_paths[-1]] * (self.max_images - len(image_paths))

        images = torch.stack([self._load_image(p) for p in image_paths], dim=0)  # [N,C,H,W]
        return images, torch.tensor(label, dtype=torch.long)
