"""
违停识别系统交互界面
用法：
python parking_system_gui.py
"""

import argparse
import json
import os
import sys
import threading
from pathlib import Path
from typing import Optional

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:
    print("错误：需要 tkinter，请安装：pip install tk")
    sys.exit(1)

try:
    from PIL import Image, ImageTk
except ImportError:
    print("错误：需要 Pillow，请安装：pip install Pillow")
    sys.exit(1)


class ParkingSystemGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("违停识别系统")
        self.root.geometry("1000x600")
        self.root.minsize(800, 500)
        self.root.configure(bg="#1a1a2e")

        # 状态变量
        self.video_path: Optional[Path] = None
        self.roi_path: Optional[Path] = None
        self.out_dir: Path = Path("outputs/result")
        self.ckpt_path: Path = Path("runs/fasterrcnn_vehicle_resnet_best_aliyun.pt")

        self._setup_ui()

    def _setup_ui(self):
        # 顶部标题
        title_frame = tk.Frame(self.root, bg="#16213e")
        title_frame.pack(fill=tk.X, pady=20)

        title_label = tk.Label(
            title_frame,
            text="🚗 违停识别系统",
            font=("微软雅黑", 24, "bold"),
            fg="#ffffff",
            bg="#16213e"
        )
        title_label.pack(pady=10)

        # 主内容区
        main_frame = tk.Frame(self.root, bg="#1a1a2e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # 左侧配置区
        config_frame = tk.Frame(main_frame, bg="#16213e", width=300)
        config_frame.pack(fill=tk.Y, side=tk.LEFT, padx=10, pady=10)
        config_frame.pack_propagate(False)

        # 视频选择
        video_frame = tk.Frame(config_frame, bg="#16213e")
        video_frame.pack(fill=tk.X, pady=10, padx=10)

        tk.Label(
            video_frame,
            text="视频文件",
            font=("微软雅黑", 12),
            fg="#a0a0c0",
            bg="#16213e"
        ).pack(anchor="w", pady=5)

        self.video_var = tk.StringVar(value="未选择视频")
        video_entry = tk.Entry(
            video_frame,
            textvariable=self.video_var,
            font=("微软雅黑", 10),
            bg="#0f3460",
            fg="#e0e0e0",
            borderwidth=0,
            state="readonly"
        )
        video_entry.pack(fill=tk.X, padx=5, pady=5)

        browse_btn = tk.Button(
            video_frame,
            text="浏览...",
            command=self._browse_video,
            font=("微软雅黑", 10),
            bg="#533483",
            fg="white",
            relief="flat",
            cursor="hand2"
        )
        browse_btn.pack(side=tk.RIGHT, padx=5)

        # ROI文件
        roi_frame = tk.Frame(config_frame, bg="#16213e")
        roi_frame.pack(fill=tk.X, pady=10, padx=10)

        tk.Label(
            roi_frame,
            text="ROI文件",
            font=("微软雅黑", 12),
            fg="#a0a0c0",
            bg="#16213e"
        ).pack(anchor="w", pady=5)

        self.roi_var = tk.StringVar(value="未生成ROI")
        roi_entry = tk.Entry(
            roi_frame,
            textvariable=self.roi_var,
            font=("微软雅黑", 10),
            bg="#0f3460",
            fg="#e0e0e0",
            borderwidth=0,
            state="readonly"
        )
        roi_entry.pack(fill=tk.X, padx=5, pady=5)

        # 操作按钮
        btn_frame = tk.Frame(config_frame, bg="#16213e")
        btn_frame.pack(fill=tk.X, pady=20, padx=10)

        self.annotate_btn = tk.Button(
            btn_frame,
            text="📏 标注ROI区域",
            command=self._start_roi_annotation,
            font=("微软雅黑", 11),
            bg="#4ECDC4",
            fg="#1a1a2e",
            relief="flat",
            cursor="hand2",
            state=tk.DISABLED
        )
        self.annotate_btn.pack(fill=tk.X, pady=5, padx=5)

        self.detect_btn = tk.Button(
            btn_frame,
            text="🚨 开始检测",
            command=self._start_detection,
            font=("微软雅黑", 11),
            bg="#FF6B6B",
            fg="white",
            relief="flat",
            cursor="hand2",
            state=tk.DISABLED
        )
        self.detect_btn.pack(fill=tk.X, pady=5, padx=5)

        self.view_btn = tk.Button(
            btn_frame,
            text="🎬 查看结果视频",
            command=self._view_result,
            font=("微软雅黑", 11),
            bg="#27ae60",
            fg="white",
            relief="flat",
            cursor="hand2",
            state=tk.DISABLED
        )
        self.view_btn.pack(fill=tk.X, pady=5, padx=5)

        # 右侧信息区
        info_frame = tk.Frame(main_frame, bg="#16213e")
        info_frame.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=10, pady=10)

        tk.Label(
            info_frame,
            text="系统状态",
            font=("微软雅黑", 14, "bold"),
            fg="#a0a0c0",
            bg="#16213e"
        ).pack(anchor="w", pady=10, padx=10)

        self.status_text = tk.Text(
            info_frame,
            font=("微软雅黑", 10),
            bg="#0f3460",
            fg="#e0e0e0",
            borderwidth=0,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 底部状态栏
        status_frame = tk.Frame(self.root, bg="#16213e")
        status_frame.pack(fill=tk.X, pady=10)

        self.status_var = tk.StringVar(value="就绪 | 请选择视频文件")
        status_label = tk.Label(
            status_frame,
            textvariable=self.status_var,
            font=("微软雅黑", 10),
            fg="#e0e0e0",
            bg="#16213e"
        )
        status_label.pack(anchor="w", padx=20, pady=5)

    def _browse_video(self):
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.wmv")]
        )
        if file_path:
            self.video_path = Path(file_path)
            self.video_var.set(str(self.video_path))
            # 自动生成ROI路径
            self.roi_path = Path("configs") / f"{self.video_path.stem}_roi.json"
            self.roi_var.set(str(self.roi_path))
            self.annotate_btn.config(state=tk.NORMAL)
            self._update_status(f"已选择视频: {self.video_path.name}")

    def _update_status(self, message):
        self.status_var.set(message)
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)

    def _start_roi_annotation(self):
        if not self.video_path:
            messagebox.showwarning("提示", "请先选择视频文件")
            return

        # 启动ROI标注工具
        self._update_status("正在启动ROI标注工具...")
        
        def run_annotation():
            try:
                # 使用子进程运行ROI标注工具
                import subprocess
                
                # 构造命令
                cmd = [
                    sys.executable,
                    str(Path(__file__).parent / "roi_writer.py"),
                    "--video", str(self.video_path),
                    "--out", str(self.roi_path)
                ]
                
                # 运行标注工具
                subprocess.run(cmd, check=True)
                
                # 标注完成后更新状态
                if self.roi_path.exists():
                    self._update_status("ROI标注完成！")
                    self.detect_btn.config(state=tk.NORMAL)
                else:
                    self._update_status("ROI标注未完成")
            except Exception as e:
                self._update_status(f"标注过程出错: {str(e)}")

        # 在新线程中运行，避免阻塞UI
        threading.Thread(target=run_annotation, daemon=True).start()

    def _start_detection(self):
        if not self.video_path or not self.roi_path or not self.roi_path.exists():
            messagebox.showwarning("提示", "请先完成ROI标注")
            return

        # 启动违停检测
        self._update_status("正在启动违停检测...")
        
        def run_detection():
            try:
                # 使用子进程运行违停检测
                import subprocess
                
                # 构造命令
                cmd = [
                    sys.executable,
                    str(Path(__file__).parent / "parking_violation.py"),
                    "--video", str(self.video_path),
                    "--roi", str(self.roi_path),
                    "--ckpt", str(self.ckpt_path),
                    "--out_dir", str(self.out_dir)
                ]
                
                # 运行检测
                subprocess.run(cmd, check=True)
                
                # 检测完成后更新状态
                self._update_status("违停检测完成！")
                self.view_btn.config(state=tk.NORMAL)
            except Exception as e:
                self._update_status(f"检测过程出错: {str(e)}")

        # 在新线程中运行，避免阻塞UI
        threading.Thread(target=run_detection, daemon=True).start()

    def _view_result(self):
        # 查看结果视频
        result_video = self.out_dir / "overlay.mp4"
        if not result_video.exists():
            messagebox.showwarning("提示", "结果视频不存在")
            return

        self._update_status(f"正在打开结果视频: {result_video}")
        
        # 使用系统默认播放器打开视频
        try:
            if os.name == 'nt':  # Windows
                os.startfile(result_video)
            else:  # Linux/macOS
                import subprocess
                subprocess.run(['xdg-open', result_video])
        except Exception as e:
            self._update_status(f"打开视频失败: {str(e)}")


def main():
    root = tk.Tk()
    app = ParkingSystemGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()