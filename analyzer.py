from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import base64
import math

import cv2
import numpy as np

REQUIRED_ANGLES = ["front", "up", "down", "left", "right"]

FRONTAL_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
PROFILE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def _decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("ไม่สามารถอ่านรูปภาพได้")
    return img

def _largest_rect(rects):
    if len(rects) == 0:
        return None
    rects = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
    return tuple(int(v) for v in rects[0])

def _detect_face(img_bgr: np.ndarray) -> Tuple[Optional[Tuple[int,int,int,int]], str]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = FRONTAL_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    best = _largest_rect(faces)
    if best is not None:
        return best, "frontal"

    # Try profile detection in original and mirrored image
    prof = PROFILE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    best = _largest_rect(prof)
    if best is not None:
        return best, "profile"

    flipped = cv2.flip(gray, 1)
    prof2 = PROFILE_CASCADE.detectMultiScale(flipped, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    best2 = _largest_rect(prof2)
    if best2 is not None:
        x, y, w, h = best2
        # map to original coordinates
        x2 = gray.shape[1] - x - w
        return (int(x2), int(y), int(w), int(h)), "profile_mirrored"

    return None, "none"

def _crop(img, rect, pad=0.12):
    h, w = img.shape[:2]
    x, y, bw, bh = rect
    px = int(bw * pad)
    py = int(bh * pad)
    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(w, x + bw + px)
    y2 = min(h, y + bh + py)
    return img[y1:y2, x1:x2].copy()

def _symmetry_score(face_bgr: np.ndarray) -> float:
    if face_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (160, 160))
    left = gray[:, :80]
    right = cv2.flip(gray[:, 80:], 1)
    diff = cv2.absdiff(left, right)
    score = 1.0 - (float(diff.mean()) / 255.0)
    return float(np.clip(score, 0.0, 1.0))

def _sharpness_score(face_bgr: np.ndarray) -> float:
    if face_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize reasonably for phone photos
    score = min(var / 220.0, 1.0)
    return float(np.clip(score, 0.0, 1.0))

def _center_score(img_shape, rect) -> float:
    h, w = img_shape[:2]
    x, y, bw, bh = rect
    cx = x + bw / 2
    cy = y + bh / 2
    dx = abs(cx - w / 2) / (w / 2)
    dy = abs(cy - h / 2) / (h / 2)
    score = 1.0 - min((dx * 0.55 + dy * 0.45), 1.0)
    return float(np.clip(score, 0.0, 1.0))

def _size_score(img_shape, rect) -> float:
    h, w = img_shape[:2]
    x, y, bw, bh = rect
    area_ratio = (bw * bh) / float(w * h)
    # Favor medium/close face area
    if area_ratio < 0.03:
        return max(0.0, area_ratio / 0.03)
    if area_ratio > 0.45:
        return max(0.0, 1.0 - ((area_ratio - 0.45) / 0.55))
    # Peak around 0.12~0.25
    target = 0.18
    score = 1.0 - abs(area_ratio - target) / target
    return float(np.clip(score, 0.0, 1.0))

def _eye_count_score(face_bgr: np.ndarray) -> float:
    if face_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    eyes = EYE_CASCADE.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=7, minSize=(18, 18))
    n = len(eyes)
    if n >= 2:
        return 1.0
    if n == 1:
        return 0.55
    return 0.15

def _orientation_score(label: str, face_kind: str, img_shape, rect, face_bgr: np.ndarray) -> Tuple[float, dict]:
    """
    Baseline heuristic score for each requested angle.
    This is NOT a medical/biometric grade verifier; it is a starter scoring model.
    """
    x, y, bw, bh = rect
    h, w = img_shape[:2]
    aspect = bw / max(1.0, bh)
    area_ratio = (bw * bh) / float(w * h)
    center = _center_score(img_shape, rect)
    size = _size_score(img_shape, rect)
    sharp = _sharpness_score(face_bgr)
    sym = _symmetry_score(face_bgr)
    eyes = _eye_count_score(face_bgr)

    # Shared base
    base = 0.28 + 0.18 * center + 0.17 * size + 0.17 * sharp + 0.20 * eyes

    debug = {
        "center_score": round(center, 4),
        "size_score": round(size, 4),
        "sharpness_score": round(sharp, 4),
        "symmetry_score": round(sym, 4),
        "eyes_score": round(eyes, 4),
        "face_kind": face_kind,
        "face_aspect_ratio": round(float(aspect), 4),
        "face_area_ratio": round(float(area_ratio), 4),
    }

    label = label.lower().strip()

    if label == "front":
        # frontal face should be centered and symmetric
        score = base + 0.22 * sym + (0.08 if face_kind.startswith("frontal") else -0.05)
    elif label in ("left", "right"):
        # profile-like images often have less symmetry and face detector may be profile
        profile_bonus = 0.20 if "profile" in face_kind else 0.00
        side_balance = 1.0 - min(abs(aspect - 0.88) / 0.88, 1.0)
        score = base + profile_bonus + 0.10 * (1.0 - sym) + 0.10 * side_balance
    elif label in ("up", "down"):
        # look for a face that is present with medium-size and not too wide
        vertical_hint = 1.0 - min(abs(aspect - 0.80) / 0.80, 1.0)
        score = base + 0.12 * vertical_hint + 0.08 * (1.0 - abs(center - 0.50))
    else:
        score = base

    # If no face kind at all, heavily penalize
    if face_kind == "none":
        score = 0.0

    score = float(np.clip(score, 0.0, 1.0))
    return score, debug

def _image_to_b64_preview(img_bgr: np.ndarray, max_w=480) -> str:
    h, w = img_bgr.shape[:2]
    if w > max_w:
        scale = max_w / w
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def analyze_single(image_bytes: bytes, label: str) -> dict:
    img = _decode_image(image_bytes)
    rect, face_kind = _detect_face(img)

    if rect is None:
        preview = _image_to_b64_preview(img)
        return {
            "label": label,
            "face_detected": False,
            "score": 0.0,
            "status": "no_face",
            "face_kind": "none",
            "debug": {},
            "preview": preview,
        }

    face = _crop(img, rect, pad=0.12)
    score, debug = _orientation_score(label, face_kind, img.shape, rect, face)

    # Convert to a percentage with a gentle floor for valid detections
    percent = round(score * 100.0, 2)

    preview = _image_to_b64_preview(img)
    return {
        "label": label,
        "face_detected": True,
        "score": percent,
        "status": "ok" if percent >= 55 else "weak",
        "face_kind": face_kind,
        "debug": debug,
        "preview": preview,
    }

def analyze_face_set(payload: Dict[str, Optional[bytes]]) -> dict:
    results = []
    present = 0
    scores = []

    for angle in REQUIRED_ANGLES:
        img_bytes = payload.get(angle)
        if not img_bytes:
            results.append({
                "label": angle,
                "face_detected": False,
                "score": 0.0,
                "status": "missing",
                "face_kind": "none",
                "debug": {},
                "preview": "",
            })
            continue

        item = analyze_single(img_bytes, angle)
        results.append(item)
        if item["face_detected"]:
            present += 1
            scores.append(float(item["score"]))

    completeness = round((present / len(REQUIRED_ANGLES)) * 100.0, 2)
    mean_score = round((sum(scores) / len(scores)) if scores else 0.0, 2)

    # overall probability: weighted combination of completeness and mean face score
    overall = round(min(100.0, max(0.0, (0.55 * completeness) + (0.45 * mean_score))), 2)

    status = "Human-likely" if overall >= 70 and completeness >= 80 else "Needs review" if overall >= 45 else "Unclear"

    return {
        "overall": overall,
        "completeness": completeness,
        "mean_score": mean_score,
        "status": status,
        "required_angles": REQUIRED_ANGLES,
        "results": results,
        "note": "This is a baseline image-analysis scorer (face detection + quality + pose heuristics)."
    }
