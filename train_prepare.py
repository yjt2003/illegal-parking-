import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd: Path | None = None) -> None:
    print(f"[run] {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None)
    if result.returncode != 0:
        raise SystemExit(f"command failed with code {result.returncode}: {' '.join(str(c) for c in cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="一键执行：从 COCO labels.json 生成 vehicle 单类检测数据集 + 训练 Faster R-CNN(ResNet50-FPN)。"
    )
    ap.add_argument(
        "--project_root",
        type=Path,
        default=Path(r"d:\毕业设计"),
        help="项目根目录（包含 scripts/、coco-2017/ 等）。默认 d:\\毕业设计。",
    )
    ap.add_argument(
        "--labels",
        type=Path,
        default=Path(r"d:\毕业设计\coco-2017\train\labels.json"),
        help="原始 COCO 检测标注 labels.json 路径。",
    )
    ap.add_argument(
        "--images_dir",
        type=Path,
        default=Path(r"d:\毕业设计\coco-2017\train\data"),
        help="COCO 图片目录（与 labels.json 中 file_name 对应）。",
    )
    ap.add_argument(
        "--det_out_dir",
        type=Path,
        default=Path(r"d:\毕业设计\data\vehicle_det_coco1"),
        help="单类 vehicle 检测数据集 JSON 输出目录（train.json / val.json）。",
    )
    ap.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="验证集占比（按图片划分），默认 0.2。",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Faster R-CNN 训练轮数，默认 10。",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="训练 batch size，默认 2（显存不够可以改小）。",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Faster R-CNN 学习率，默认 5e-4。",
    )
    ap.add_argument(
        "--no_pretrained",
        action="store_true",
        help="不使用 COCO 预训练权重（默认使用预训练）。",
    )
    args = ap.parse_args()

    project_root = args.project_root
    project_root.mkdir(parents=True, exist_ok=True)

    python_exe = sys.executable  # 使用当前 Python 解释器

    # 1) 跳过数据准备步骤，因为scripts目录已删除
    # 假设数据集已经准备好
    args.det_out_dir.mkdir(parents=True, exist_ok=True)
    print("跳过数据准备步骤，使用已有的数据集")

    # 2) 训练 Faster R-CNN(ResNet50-FPN) 车辆检测器
    train_script = project_root / "train_model.py"
    if not train_script.is_file():
        raise SystemExit(f"train script not found: {train_script}")

    train_cmd = [
        python_exe,
        str(train_script),
        "--train_json",
        str(args.det_out_dir / "train.json"),
        "--val_json",
        str(args.det_out_dir / "val.json"),
        "--images_dir",
        str(args.images_dir),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
    ]
    if not args.no_pretrained:
        train_cmd.append("--pretrained")

    run(train_cmd, cwd=project_root)

    print("\n全部流程完成：")
    print(f"- 单类 vehicle COCO 检测 JSON：{args.det_out_dir}")
    print("- Faster R-CNN 训练输出目录：runs/fasterrcnn_vehicle_resnet")


if __name__ == "__main__":
    main()

