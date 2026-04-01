import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm


@dataclass
class EpochResult:
    loss: float
    recall: float


class CocoVehicleDetectionDataset(Dataset):
    """
    简单的 COCO 风格单类(vehicle)检测数据集，不依赖 pycocotools。
    约定 JSON 结构类似 scripts/prepare_vehicle_det_coco1.py 生成的 train.json/val.json。
    """

    def __init__(self, json_path: Path, images_dir: Path):
        self.json_path = Path(json_path)
        self.images_dir = Path(images_dir)

        with self.json_path.open("r", encoding="utf-8") as f:
            coco: Dict[str, Any] = json.load(f)

        self.images: List[Dict[str, Any]] = coco.get("images", [])
        anns: List[Dict[str, Any]] = coco.get("annotations", [])

        image_to_anns: Dict[int, List[Dict[str, Any]]] = {}
        for a in anns:
            img_id = int(a["image_id"])
            image_to_anns.setdefault(img_id, []).append(a)

        self.image_to_anns = image_to_anns

        # 只保留至少有一个标注的图片
        self.images = [img for img in self.images if img.get("id") in self.image_to_anns]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_info = self.images[idx]
        img_id = int(img_info["id"])
        file_name = img_info["file_name"]

        img_path = self.images_dir / file_name
        img = Image.open(img_path).convert("RGB")

        anns = self.image_to_anns.get(img_id, [])

        boxes: List[List[float]] = []
        for a in anns:
            x, y, w, h = a["bbox"]
            x2 = x + w
            y2 = y + h
            boxes.append([x, y, x2, y2])

        num_objs = len(boxes)

        if num_objs == 0:
            # 理论上 prepare_vehicle_det_coco1 已过滤掉无标注图片，这里仅作兜底
            boxes = [[0.0, 0.0, 1.0, 1.0]]
            labels = [1]
        else:
            labels = [1] * num_objs  # 单类 vehicle -> id 1

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        image_id_tensor = torch.as_tensor([img_id], dtype=torch.int64)
        area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        iscrowd = torch.zeros((boxes_tensor.shape[0],), dtype=torch.int64)

        # torchvision 检测模型期望的 target 格式
        target: Dict[str, torch.Tensor] = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": image_id_tensor,
            "area": area,
            "iscrowd": iscrowd,
        }

        # 不做几何增强，交给模型内部的 GeneralizedRCNNTransform 负责 resize
        img_arr = np.array(img, dtype=np.uint8)
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float() / 255.0

        return img_tensor, target


def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    计算 IoU，boxes: [N, 4] in (x1, y1, x2, y2)
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    x11, y11, x12, y12 = boxes1.unbind(1)
    x21, y21, x22, y22 = boxes2.unbind(1)

    inter_x1 = torch.max(x11[:, None], x21[None, :])
    inter_y1 = torch.max(y11[:, None], y21[None, :])
    inter_x2 = torch.min(x12[:, None], x22[None, :])
    inter_y2 = torch.min(y12[:, None], y22[None, :])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)

    union = area1[:, None] + area2[None, :] - inter_area
    iou = inter_area / union.clamp(min=1e-6)
    return iou


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_thresh: float = 0.5,
    iou_thresh: float = 0.5,
) -> EpochResult:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_gt = 0
    matched_gt = 0

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="val", leave=False):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            # 这里只做简单的检测性能评估，不再重新计算损失，
            # 避免在不同 torchvision 版本下 model(images, targets) 行为差异导致错误。
            loss = 0.0
            bs = len(images)
            total_loss += loss * bs
            total_samples += bs

            # 简单 Recall 估计：在给定阈值下，每个 GT 是否被至少一个预测框覆盖
            for out, tgt in zip(outputs, targets):
                gt_boxes = tgt["boxes"]
                total_gt += int(gt_boxes.shape[0])

                if gt_boxes.numel() == 0:
                    continue

                pred_boxes = out["boxes"]
                scores = out["scores"]
                keep = scores >= score_thresh
                pred_boxes = pred_boxes[keep]

                if pred_boxes.numel() == 0:
                    continue

                ious = box_iou(gt_boxes, pred_boxes)
                # 贪心匹配：每个 GT 若存在 IoU>=阈值的预测则记为命中
                gt_max_iou, _ = ious.max(dim=1)
                matched_gt += int((gt_max_iou >= iou_thresh).sum().item())

    avg_loss = total_loss / max(1, total_samples)
    recall = matched_gt / max(1, total_gt)
    return EpochResult(loss=avg_loss, recall=recall)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    n_batches = len(loader)

    for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="train", leave=False)):
        try:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            optimizer.step()

            bs = len(images)
            total_loss += float(loss.item()) * bs
            total_samples += bs
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(
                    f"\n[错误] 训练在 batch {batch_idx + 1}/{n_batches} 中断，疑似显存/内存不足 (OOM)。\n"
                    "建议：--batch_size 1 --min_size 640 --max_size 1000 或 --num_workers 0"
                )
            raise
        except Exception as e:
            print(f"\n[错误] 训练在 batch {batch_idx + 1}/{n_batches} 中断: {e}")
            raise

    return total_loss / max(1, total_samples)


def build_model(num_classes: int, pretrained: bool, min_size: int, max_size: int) -> torch.nn.Module:
    if pretrained:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    else:
        weights = None

    model = fasterrcnn_resnet50_fpn(weights=weights, box_detections_per_img=100, min_size=min_size, max_size=max_size)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Faster R-CNN (ResNet50-FPN) vehicle detector on single-class COCO JSON")
    ap.add_argument(
        "--train_json",
        type=Path,
        default=Path("data/vehicle_det_coco1/train.json"),
        help="Path to train COCO-style JSON (single-class vehicle).",
    )
    ap.add_argument(
        "--val_json",
        type=Path,
        default=Path("data/vehicle_det_coco1/val.json"),
        help="Path to val COCO-style JSON (single-class vehicle).",
    )
    ap.add_argument(
        "--images_dir",
        type=Path,
        default=Path("coco-2017/train/images"),
        help="Root directory of COCO images (relative paths as in JSON).",
    )
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--pretrained", action="store_true", help="Use COCO-pretrained Faster R-CNN weights")
    ap.add_argument("--min_size", type=int, default=800, help="Shorter side min size for the internal transform")
    ap.add_argument("--max_size", type=int, default=1333, help="Longer side max size for the internal transform")
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("runs/fasterrcnn_vehicle_resnet"),
        help="Output directory for checkpoints and history.",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CocoVehicleDetectionDataset(args.train_json, args.images_dir)
    val_ds = CocoVehicleDetectionDataset(args.val_json, args.images_dir)

    if len(train_ds) == 0:
        raise SystemExit(f"train dataset empty: {args.train_json}")

    num_classes = 2  # 背景 + vehicle

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(num_classes=num_classes, pretrained=args.pretrained, min_size=args.min_size, max_size=args.max_size)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)

    best_recall = -1.0
    history: List[Dict[str, Any]] = []
    last_epoch = 0

    try:
        for epoch in range(1, args.epochs + 1):
            last_epoch = epoch
            train_loss = train_one_epoch(model, train_loader, device, optimizer)
            val_res = evaluate(model, val_loader, device)

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_res.loss,
                    "val_recall": val_res.recall,
                }
            )
            (args.out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "num_classes": num_classes,
                "train_json": str(args.train_json),
                "val_json": str(args.val_json),
                "images_dir": str(args.images_dir),
                "min_size": args.min_size,
                "max_size": args.max_size,
            }
            torch.save(ckpt, args.out_dir / "last.pt")

            print(
                f"epoch {epoch:03d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_res.loss:.4f} | "
                f"val_recall={val_res.recall:.4f}"
            )

            if val_res.recall > best_recall:
                best_recall = val_res.recall
                torch.save(ckpt, args.out_dir / "best.pt")

        print(
            f"done: best_val_recall={best_recall:.4f} | "
            f"out_dir={args.out_dir} | device={device}"
        )
    except Exception as e:
        # 中断时尽量保存当前权重（可能只训练到一半 epoch）
        ckpt = {
            "epoch": last_epoch,
            "model_state": model.state_dict(),
            "num_classes": num_classes,
            "train_json": str(args.train_json),
            "val_json": str(args.val_json),
            "images_dir": str(args.images_dir),
            "min_size": args.min_size,
            "max_size": args.max_size,
        }
        torch.save(ckpt, args.out_dir / "last.pt")
        print(f"\n已保存当前权重到 {args.out_dir / 'last.pt'}，可减小 batch_size/min_size 后从该权重继续或重新训练。")
        raise


if __name__ == "__main__":
    main()

