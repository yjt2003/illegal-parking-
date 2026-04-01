"""
批量车辆检测和标注脚本

该脚本使用训练好的车辆检测模型对指定目录下的所有图像进行检测和标注。

使用方法：
python detection_batch.py --data_dir <数据集目录> --out_dir <输出目录>

参数说明：
--data_dir: 包含要处理图像的目录
--out_dir: 保存检测结果和标注的目录
--ckpt: 模型权重文件路径（默认：runs/fasterrcnn_vehicle_resnet/best.pt）
--score_thresh: 检测阈值（默认：0.5）
--cpu: 强制使用CPU进行推理
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def build_model(num_classes: int, min_size: int = 800, max_size: int = 1333) -> torch.nn.Module:
    """构建车辆检测模型"""
    model = fasterrcnn_resnet50_fpn(weights=None, box_detections_per_img=200, min_size=min_size, max_size=max_size)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def to_tensor_rgb01(img: Image.Image) -> torch.Tensor:
    """将图像转换为张量"""
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

def draw_boxes(
    img: Image.Image,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    score_thresh: float,
) -> Tuple[Image.Image, Dict]:
    """绘制检测框并生成标注信息"""
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
    """主函数"""
    ap = argparse.ArgumentParser(description="Batch vehicle detection and annotation.")
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=Path("runs/fasterrcnn_vehicle_resnet/best.pt"),
        help="Path to model checkpoint file.",
    )
    ap.add_argument("--data_dir", type=Path, required=True, help="Directory containing images to process.")
    ap.add_argument("--out_dir", type=Path, default=Path("outputs"), help="Directory to save results.")
    ap.add_argument("--score_thresh", type=float, default=0.5, help="Score threshold for detection.")
    ap.add_argument("--min_size", type=int, default=800, help="Model input min size.")
    ap.add_argument("--max_size", type=int, default=1333, help="Model input max size.")
    ap.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    args = ap.parse_args()

    # 设置设备
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    # 创建输出目录
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.out_dir}")

    # 加载模型
    print(f"Loading model from: {args.ckpt}")
    try:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        num_classes = int(ckpt.get("num_classes", 2))
        print(f"Model loaded successfully. Number of classes: {num_classes}")

        model = build_model(num_classes=num_classes, min_size=args.min_size, max_size=args.max_size)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()
        print("Model ready for inference")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 支持的图像格式
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # 收集所有图像文件
    image_files = []
    for ext in supported_formats:
        found = list(args.data_dir.glob(f"**/*{ext}"))
        image_files.extend(found)
    
    total_images = len(image_files)
    print(f"Found {total_images} images to process")
    
    if total_images == 0:
        print("No images found. Exiting.")
        return

    # 汇总标注结果
    all_annotations = {}
    processed_count = 0
    error_count = 0
    
    # 处理每个图像
    for idx, img_path in enumerate(image_files, 1):
        try:
            # 加载图像
            img = Image.open(img_path).convert("RGB")
            x = to_tensor_rgb01(img).to(device)

            # 模型推理
            with torch.no_grad():
                out = model([x])[0]

            boxes = out["boxes"]
            scores = out["scores"]

            # 绘制检测结果
            vis, meta = draw_boxes(img, boxes, scores, score_thresh=args.score_thresh)

            # 保存结果
            rel_path = img_path.relative_to(args.data_dir)
            out_subdir = args.out_dir / rel_path.parent
            out_subdir.mkdir(parents=True, exist_ok=True)
            
            out_img = out_subdir / f"{img_path.stem}_det.jpg"
            out_json = out_subdir / f"{img_path.stem}_det.json"
            
            vis.save(out_img, quality=95)
            out_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            
            # 添加到汇总标注
            all_annotations[str(rel_path)] = meta
            
            processed_count += 1
            if idx % 10 == 0:
                print(f"Processed {idx}/{total_images} images")
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            error_count += 1
    
    # 保存汇总标注文件
    summary_json = args.out_dir / "annotations_summary.json"
    summary_json.write_text(json.dumps(all_annotations, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print(f"\nProcessing complete!")
    print(f"Total images: {total_images}")
    print(f"Processed: {processed_count}")
    print(f"Errors: {error_count}")
    print(f"Output directory: {args.out_dir}")
    print(f"Summary annotations saved to: {summary_json}")

if __name__ == "__main__":
    main()
