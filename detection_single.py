import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(num_classes: int, min_size: int = 800, max_size: int = 1333) -> torch.nn.Module:
    # 推理时不需要预训练权重；我们会加载自己的 ckpt
    model = fasterrcnn_resnet50_fpn(weights=None, box_detections_per_img=200, min_size=min_size, max_size=max_size)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def to_tensor_rgb01(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


def draw_boxes(
    img: Image.Image,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    score_thresh: float,
) -> Tuple[Image.Image, Dict]:
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    kept = scores >= score_thresh
    boxes = boxes[kept].cpu()
    scores = scores[kept].cpu()

    dets = []
    for i, (b, s) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = [float(v) for v in b.tolist()]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        txt = f"vehicle {float(s):.2f}"
        tx, ty = x1, max(0.0, y1 - 12)
        if font is not None:
            draw.text((tx, ty), txt, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        else:
            draw.text((tx, ty), txt, fill=(255, 255, 255))

        dets.append({"box_xyxy": [x1, y1, x2, y2], "score": float(s)})

    meta = {"num_dets": len(dets), "score_thresh": score_thresh, "dets": dets}
    return img, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Load Faster R-CNN vehicle detector checkpoint and run inference on one image.")
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=Path("runs/fasterrcnn_vehicle_resnet/best.pt"),
        help="Path to checkpoint produced by train_model.py (best.pt/last.pt).",
    )
    ap.add_argument("--image", type=Path, required=True, help="Path to a local image file (.jpg/.png).")
    ap.add_argument("--out_dir", type=Path, default=Path("outputs"), help="Directory to save visualized result.")
    ap.add_argument("--score_thresh", type=float, default=0.5, help="Score threshold for visualization.")
    ap.add_argument("--min_size", type=int, default=800, help="Model transform min_size (should match training if possible).")
    ap.add_argument("--max_size", type=int, default=1333, help="Model transform max_size (should match training if possible).")
    ap.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    num_classes = int(ckpt.get("num_classes", 2))

    model = build_model(num_classes=num_classes, min_size=args.min_size, max_size=args.max_size)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    img = Image.open(args.image).convert("RGB")
    x = to_tensor_rgb01(img).to(device)

    with torch.no_grad():
        out = model([x])[0]

    boxes = out["boxes"]
    scores = out["scores"]

    vis, meta = draw_boxes(img, boxes, scores, score_thresh=args.score_thresh)

    out_img = args.out_dir / f"{args.image.stem}_det.jpg"
    out_json = args.out_dir / f"{args.image.stem}_det.json"
    vis.save(out_img, quality=95)
    out_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved: {out_img}")
    print(f"saved: {out_json}")
    print(f"device: {device} | num_dets={meta['num_dets']} | score_thresh={args.score_thresh}")


if __name__ == "__main__":
    main()

