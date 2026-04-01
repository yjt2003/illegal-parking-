# 违停识别系统

## 系统功能

- 视频文件选择
- ROI（停车区域）标注
- 违停检测
- 结果视频查看

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行系统

```bash
python parking_system_gui.py
```

### 3. 使用步骤

1. **选择视频文件**：点击"浏览..."按钮选择要分析的视频
2. **标注ROI区域**：点击"标注ROI区域"按钮，在弹出的标注工具中：
   - 左键点击添加多边形顶点
   - 右键点击完成当前区域（至少3个点）
   - Z键：撤销上一个点
   - C键：取消当前区域重新画
   - S键：保存并退出
   - Q键：退出不保存
   - 滚轮：缩放图片
3. **开始检测**：标注完成后，点击"开始检测"按钮
4. **查看结果**：检测完成后，点击"查看结果视频"按钮

## 项目结构

- `parking_system_gui.py`：主交互界面
- `roi_writer.py`：ROI标注工具
- `parking_violation.py`：违停检测核心算法
- `detection_single.py`：目标检测模型
- `configs/`：存储ROI配置文件
- `outputs/`：存储检测结果
- `runs/`：存储模型权重

## 技术说明

1. **目标检测**：使用Faster R-CNN模型检测车辆
2. **跟踪**：基于IoU的多目标跟踪
3. **ROI判定**：使用射线法判断车辆是否在停车位内
4. **违停判定**：基于停留时间阈值
5. **可视化**：在视频上叠加检测结果和ROI区域

## 配置说明

- 违停判定阈值：默认6秒
- 丢检容忍时间：默认1.5秒
- 检测分数阈值：默认0.5
- 模型权重路径：`runs/fasterrcnn_vehicle_resnet_best_aliyun.pt`
- 输出目录：`outputs/result`

## 注意事项

1. 确保视频文件格式支持（.mp4, .avi, .mov, .wmv）
2. 标注ROI时，确保每个停车位都被正确标注
3. 检测过程可能需要较长时间，取决于视频长度和硬件性能
4. 结果视频将保存在`outputs/result/overlay.mp4`

## 常见问题

### Q: 运行时提示缺少模块
A: 请确保已安装所有依赖：`pip install -r requirements.txt`

### Q: 标注工具无法启动
A: 确保已安装tkinter：`pip install tk`

### Q: 检测过程中出现错误
A: 检查视频文件是否损坏，或模型权重文件是否存在

### Q: 结果视频没有声音
A: 系统会尝试保留原始视频的音频，但如果处理失败，会使用无音频的视频

## 输出文件说明

- `overlay.mp4`：带有检测结果的视频
- `events.json`：违停事件详情
- `events.csv`：违停事件表格
- `run_meta.json`：运行配置信息
- `snapshots/`：违停事件截图