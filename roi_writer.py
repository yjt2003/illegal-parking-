"""
ROI 停车格标注工具
用法：
  # 直接用图片
  python roi_annotator.py --image parking.jpg

  # 从视频提取第一帧作为背景
  python roi_annotator.py --video your_video.mp4

操作说明：
  左键点击  → 添加多边形顶点
  右键点击  → 完成当前区域（至少3个点）
  Z键       → 撤销上一个点
  C键       → 取消当前区域重新画
  S键       → 保存 roi.json 并退出
  Q键       → 退出不保存
  滚轮      → 缩放图片
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

try:
    import tkinter as tk
    from tkinter import messagebox, simpledialog
except ImportError:
    print("错误：需要 tkinter，请安装：pip install tk")
    sys.exit(1)

try:
    from PIL import Image, ImageTk, ImageDraw
except ImportError:
    print("错误：需要 Pillow，请安装：pip install Pillow")
    sys.exit(1)


# ===== 颜色配置 =====
COLORS = [
    "#FF6B6B",  # 红
    "#4ECDC4",  # 青
    "#45B7D1",  # 蓝
    "#96CEB4",  # 绿
    "#FFEAA7",  # 黄
    "#DDA0DD",  # 紫
    "#F0A500",  # 橙
    "#00CEC9",  # 深青
]


class ROIAnnotator:
    def __init__(self, root: tk.Tk, image: Image.Image, out_path: Path):
        self.root = root
        self.root.title("ROI 停车格标注工具 | 左键=添加点 右键=完成区域 S=保存 Q=退出")
        self.root.configure(bg="#1a1a2e")

        self.orig_image = image.convert("RGB")
        self.out_path = out_path
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # 已完成的 ROI 列表
        self.rois: List[Dict[str, Any]] = []
        # 当前正在画的点列表（屏幕坐标）
        self.current_points: List[List[float]] = []
        self.roi_counter = 1

        self._setup_ui()
        self._fit_to_window()
        self._render()

    def _setup_ui(self):
        # 顶部状态栏
        self.status_var = tk.StringVar(value="💡 左键点击添加顶点，右键完成当前区域，S 保存")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            bg="#16213e",
            fg="#e0e0e0",
            font=("Consolas", 11),
            pady=6,
            anchor="w",
            padx=12,
        )
        status_bar.pack(fill=tk.X, side=tk.TOP)

        # 左侧面板
        left_frame = tk.Frame(self.root, bg="#16213e", width=200)
        left_frame.pack(fill=tk.Y, side=tk.LEFT, padx=0)
        left_frame.pack_propagate(False)

        tk.Label(
            left_frame, text="已标注区域", bg="#16213e", fg="#a0a0c0",
            font=("Consolas", 10, "bold"), pady=8
        ).pack(fill=tk.X)

        # 区域列表
        list_frame = tk.Frame(left_frame, bg="#16213e")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=4)

        self.roi_listbox = tk.Listbox(
            list_frame,
            bg="#0f3460",
            fg="#e0e0e0",
            font=("Consolas", 10),
            selectbackground="#533483",
            borderwidth=0,
            highlightthickness=0,
        )
        self.roi_listbox.pack(fill=tk.BOTH, expand=True)

        # 底部按钮
        btn_frame = tk.Frame(left_frame, bg="#16213e", pady=8)
        btn_frame.pack(fill=tk.X, padx=4)

        btn_style = {"bg": "#533483", "fg": "white", "font": ("Consolas", 10),
                     "relief": "flat", "pady": 4, "cursor": "hand2"}

        save_style = {**btn_style,"bg": "#27ae60"}
        quit_style = {**btn_style,"bg": "#c0392b"}
        tk.Button(btn_frame, text="🗑 删除选中", command=self._delete_selected, **btn_style).pack(fill=tk.X, pady=2)
        tk.Button(btn_frame, text="💾 保存 (S)", command=self._save, **save_style).pack(fill=tk.X, pady=2)
        tk.Button(btn_frame, text="❌ 退出 (Q)", command=self._quit, **quit_style).pack(fill=tk.X, pady=2)

        # 快捷键说明
        help_text = "左键: 添加顶点\n右键: 完成区域\nZ: 撤销点\nC: 取消当前\nS: 保存\nQ: 退出\n滚轮: 缩放"
        tk.Label(
            left_frame, text=help_text, bg="#16213e", fg="#707090",
            font=("Consolas", 9), justify=tk.LEFT, padx=8, pady=8
        ).pack(fill=tk.X)

        # 画布
        canvas_frame = tk.Frame(self.root, bg="#0d0d1a")
        canvas_frame.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        self.canvas = tk.Canvas(
            canvas_frame, bg="#0d0d1a", cursor="crosshair",
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 绑定事件
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<MouseWheel>", self._on_scroll)      # Windows
        self.canvas.bind("<Button-4>", self._on_scroll)        # Linux 上滚
        self.canvas.bind("<Button-5>", self._on_scroll)        # Linux 下滚
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.root.bind("<KeyPress-s>", lambda e: self._save())
        self.root.bind("<KeyPress-S>", lambda e: self._save())
        self.root.bind("<KeyPress-q>", lambda e: self._quit())
        self.root.bind("<KeyPress-Q>", lambda e: self._quit())
        self.root.bind("<KeyPress-z>", lambda e: self._undo_point())
        self.root.bind("<KeyPress-Z>", lambda e: self._undo_point())
        self.root.bind("<KeyPress-c>", lambda e: self._cancel_current())
        self.root.bind("<KeyPress-C>", lambda e: self._cancel_current())
        self.root.bind("<Configure>", lambda e: self._render())

        self.mouse_x = 0
        self.mouse_y = 0

    def _fit_to_window(self):
        self.root.update_idletasks()
        cw = max(self.canvas.winfo_width(), 800)
        ch = max(self.canvas.winfo_height(), 600)
        iw, ih = self.orig_image.size
        self.scale = min(cw / iw, ch / ih, 1.0)
        self.offset_x = (cw - iw * self.scale) / 2
        self.offset_y = (ch - ih * self.scale) / 2

    def _screen_to_image(self, sx: float, sy: float):
        """屏幕坐标 → 图像坐标"""
        ix = (sx - self.offset_x) / self.scale
        iy = (sy - self.offset_y) / self.scale
        return ix, iy

    def _image_to_screen(self, ix: float, iy: float):
        """图像坐标 → 屏幕坐标"""
        sx = ix * self.scale + self.offset_x
        sy = iy * self.scale + self.offset_y
        return sx, sy

    def _render(self, event=None):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        iw, ih = self.orig_image.size
        disp_w = int(iw * self.scale)
        disp_h = int(ih * self.scale)

        # 缩放图片
        display_img = self.orig_image.resize((disp_w, disp_h), Image.LANCZOS)

        # 在图片上叠加已完成的 ROI
        overlay = display_img.copy().convert("RGBA")
        draw = ImageDraw.Draw(overlay)

        for i, roi in enumerate(self.rois):
            color_hex = COLORS[i % len(COLORS)]
            r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
            pts = [self._image_to_screen(p[0], p[1]) for p in roi["polygon"]]
            pts_flat = [(x - self.offset_x + (cw - disp_w) / 2,
                         y - self.offset_y + (ch - disp_h) / 2) for x, y in pts]
            # 实际上用相对于图片左上角的坐标
            pts_on_img = [(p[0] * self.scale, p[1] * self.scale) for p in roi["polygon"]]
            if len(pts_on_img) >= 2:
                draw.polygon(pts_on_img, fill=(r, g, b, 60), outline=(r, g, b, 220))
            # 画顶点
            for px, py in pts_on_img:
                draw.ellipse([px-4, py-4, px+4, py+4], fill=(r, g, b, 255))
            # 标号
            if pts_on_img:
                cx = sum(p[0] for p in pts_on_img) / len(pts_on_img)
                cy = sum(p[1] for p in pts_on_img) / len(pts_on_img)
                draw.text((cx - 6, cy - 8), str(roi["id"]), fill=(255, 255, 255, 220))

        # 当前正在画的点
        if self.current_points:
            cur_color = COLORS[len(self.rois) % len(COLORS)]
            r2, g2, b2 = int(cur_color[1:3], 16), int(cur_color[3:5], 16), int(cur_color[5:7], 16)
            cur_on_img = [(p[0] * self.scale, p[1] * self.scale) for p in self.current_points]
            if len(cur_on_img) >= 2:
                draw.line([coord for pt in cur_on_img for coord in pt] +
                          [cur_on_img[0][0], cur_on_img[0][1]],
                          fill=(r2, g2, b2, 200), width=2)
            for px, py in cur_on_img:
                draw.ellipse([px-5, py-5, px+5, py+5], fill=(r2, g2, b2, 255))

        display_img = Image.alpha_composite(overlay.convert("RGBA"),
                                            Image.new("RGBA", overlay.size, (0, 0, 0, 0)))
        display_img = display_img.convert("RGB")

        self.tk_image = ImageTk.PhotoImage(display_img)
        self.canvas.delete("all")

        ox = (cw - disp_w) // 2
        oy = (ch - disp_h) // 2
        self.offset_x = ox
        self.offset_y = oy

        self.canvas.create_image(ox, oy, anchor=tk.NW, image=self.tk_image)

        # 画当前线到鼠标
        if self.current_points:
            last = self.current_points[-1]
            lsx, lsy = self._image_to_screen(last[0], last[1])
            self.canvas.create_line(lsx, lsy, self.mouse_x, self.mouse_y,
                                    fill=COLORS[len(self.rois) % len(COLORS)],
                                    width=1, dash=(4, 4))

    def _on_left_click(self, event):
        ix, iy = self._screen_to_image(event.x, event.y)
        iw, ih = self.orig_image.size
        if ix < 0 or iy < 0 or ix > iw or iy > ih:
            return
        self.current_points.append([round(ix, 1), round(iy, 1)])
        n = len(self.current_points)
        self.status_var.set(f"✏️  已添加第 {n} 个顶点 ({ix:.0f}, {iy:.0f})  |  右键完成区域")
        self._render()

    def _on_right_click(self, event):
        if len(self.current_points) < 3:
            messagebox.showwarning("提示", "至少需要 3 个点才能构成一个区域！")
            return
        name = simpledialog.askstring(
            "命名区域",
            f"为第 {self.roi_counter} 个区域命名（直接回车使用默认名）：",
            initialvalue=f"zone_{self.roi_counter}"
        )
        if name is None:
            name = f"zone_{self.roi_counter}"

        roi = {
            "id": self.roi_counter,
            "name": name,
            "polygon": self.current_points.copy()
        }
        self.rois.append(roi)
        self.roi_listbox.insert(
            tk.END,
            f"#{self.roi_counter} {name} ({len(self.current_points)}点)"
        )
        self.roi_listbox.itemconfig(
            tk.END,
            fg=COLORS[(self.roi_counter - 1) % len(COLORS)]
        )

        self.roi_counter += 1
        self.current_points = []
        self.status_var.set(f"✅ 已完成 {len(self.rois)} 个区域 | 继续标注或按 S 保存")
        self._render()

    def _on_mouse_move(self, event):
        self.mouse_x = event.x
        self.mouse_y = event.y
        if self.current_points:
            self._render()

    def _on_scroll(self, event):
        if event.num == 4 or event.delta > 0:
            factor = 1.1
        else:
            factor = 0.9

        # 以鼠标位置为中心缩放
        ix, iy = self._screen_to_image(event.x, event.y)
        self.scale *= factor
        self.scale = max(0.1, min(self.scale, 10.0))
        # 重新计算 offset 使鼠标下的图像点不变
        iw, ih = self.orig_image.size
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        self.offset_x = event.x - ix * self.scale
        self.offset_y = event.y - iy * self.scale
        self._render()

    def _undo_point(self):
        if self.current_points:
            self.current_points.pop()
            self.status_var.set(f"↩️  撤销上一个点，当前 {len(self.current_points)} 个顶点")
            self._render()

    def _cancel_current(self):
        self.current_points = []
        self.status_var.set("🔄 已取消当前区域，重新开始")
        self._render()

    def _delete_selected(self):
        sel = self.roi_listbox.curselection()
        if not sel:
            messagebox.showinfo("提示", "请先在列表中选中要删除的区域")
            return
        idx = sel[0]
        self.rois.pop(idx)
        self.roi_listbox.delete(idx)
        self.status_var.set(f"🗑 已删除区域，剩余 {len(self.rois)} 个")
        self._render()

    def _save(self):
        if not self.rois:
            messagebox.showwarning("提示", "还没有标注任何区域！")
            return
        if self.current_points:
            if messagebox.askyesno("提示", f"当前有 {len(self.current_points)} 个未完成的点，是否丢弃并保存已完成的区域？"):
                self.current_points = []
            else:
                return

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"rois": self.rois}
        self.out_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        messagebox.showinfo(
            "保存成功",
            f"✅ 已保存 {len(self.rois)} 个区域到：\n{self.out_path}\n\n"
            f"接下来运行 parking_violation.py 时用这个文件作为 --roi 参数"
        )
        self.status_var.set(f"💾 已保存到 {self.out_path}")

    def _quit(self):
        if self.rois or self.current_points:
            if not messagebox.askyesno("确认退出", "确定退出？未保存的标注将丢失。"):
                return
        self.root.destroy()


def extract_first_frame(video_path: Path) -> Image.Image:
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("无法读取视频第一帧")
        import numpy as np
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    except ImportError:
        raise SystemExit("从视频提取帧需要 opencv：pip install opencv-python")


def main():
    ap = argparse.ArgumentParser(description="ROI 停车格标注工具")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="背景图片路径（.jpg/.png）")
    group.add_argument("--video", type=Path, help="视频路径，自动提取第一帧作为背景")
    ap.add_argument(
        "--out", type=Path,
        default=Path("configs/roi.json"),
        help="输出 JSON 路径（默认：configs/roi.json）"
    )
    args = ap.parse_args()

    if args.image:
        if not args.image.exists():
            raise SystemExit(f"图片不存在：{args.image}")
        image = Image.open(args.image).convert("RGB")
        print(f"已加载图片：{args.image}  尺寸：{image.size}")
    else:
        if not args.video.exists():
            raise SystemExit(f"视频不存在：{args.video}")
        print(f"正在从视频提取第一帧：{args.video}")
        image = extract_first_frame(args.video)
        print(f"提取成功，尺寸：{image.size}")

    root = tk.Tk()
    root.geometry("1200x800")
    root.minsize(900, 600)

    app = ROIAnnotator(root, image, args.out)
    root.mainloop()

    # 退出后提示
    if args.out.exists():
        print(f"\n✅ ROI 文件已保存：{args.out}")
        with open(args.out, encoding="utf-8") as f:
            data = json.load(f)
        print(f"共 {len(data['rois'])} 个区域：")
        for r in data["rois"]:
            print(f"  #{r['id']} {r['name']}  ({len(r['polygon'])} 个顶点)")
        print(f"\n接下来运行违停检测：")
        print(f"python parking_violation.py --video your_video.mp4 --roi {args.out} --ckpt your_model.pt")


if __name__ == "__main__":
    main()
