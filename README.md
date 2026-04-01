# 违停识别系统

## 系统功能

- 视频文件选择
- ROI（停车区域）标注
- 违停检测
- 结果视频查看

## 技术原理

### 1. 系统架构

系统采用模块化设计，分为三个主要组件：
- **交互层**：`parking_system_gui.py` 提供用户操作界面
- **标注层**：`roi_writer.py` 提供停车区域标注功能
- **检测层**：`parking_violation.py` 提供违停检测核心算法

### 2. 核心技术流程

#### 2.1 视频处理流程
1. **视频选择**：用户通过GUI选择本地视频文件
2. **ROI标注**：
   - 从视频提取第一帧作为背景
   - 用户通过鼠标标注停车区域（多边形）
   - 系统保存ROI配置到JSON文件
3. **违停检测**：
   - 加载Faster R-CNN目标检测模型
   - 逐帧检测视频中的车辆
   - 基于IoU的多目标跟踪
   - 判定车辆是否在ROI内
   - 基于停留时间判断违停
   - 生成带检测结果的视频
4. **结果输出**：
   - 生成带检测结果的视频
   - 保存违停事件到JSON和CSV文件
   - 保存违停截图

#### 2.2 Faster R-CNN 目标检测

Faster R-CNN 是一种两阶段目标检测算法，其核心架构包括：

1. **骨干网络**：使用ResNet50 + FPN（特征金字塔网络）提取图像特征
2. **区域提议网络 (RPN)**：生成可能包含目标的区域提议
3. **ROI Pooling**：将不同大小的区域提议统一为固定大小的特征图
4. **分类与回归头**：对每个区域提议进行分类和边界框回归

**训练流程**：
- 使用COCO预训练权重进行迁移学习
- 多任务学习：同时优化分类和回归损失
- 早停策略：保存验证集性能最好的模型

**技术特点**：
- 高精度：两阶段检测算法，检测精度高
- 端到端训练：RPN和检测网络共享特征，端到端训练
- 灵活性：可以通过更换骨干网络来平衡精度和速度

### 2.3 多目标跟踪

- 基于IoU的贪心匹配算法
- 支持短时丢检容忍（默认1.5秒）
- 为每个车辆分配唯一的track_id

### 2.4 ROI判定

- 使用射线法判断车辆中心点是否在停车位内
- 支持多边形停车位标注

### 2.5 违停判定

- 基于停留时间阈值（默认6秒）
- 支持车辆在ROI内的停留时间累计

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
- `train_model.py`：Faster R-CNN模型训练脚本
- `train_prepare.py`：训练准备脚本
- `configs/`：存储ROI配置文件
- `outputs/`：存储检测结果
- `runs/`：存储模型权重
- `data/`：存储训练数据集

## 模型训练

### 1. 数据准备

系统使用COCO格式的标注文件进行训练，需要准备：
- 训练图像目录
- COCO格式的train.json和val.json标注文件

### 2. 训练命令

```bash
# 一键训练
python train_prepare.py --project_root d:\毕业设计 --labels d:\毕业设计\coco-2017\train\labels.json --images_dir d:\毕业设计\coco-2017\train\data

# 直接训练
python train_model.py --train_json data/vehicle_det_coco1/train.json --val_json data/vehicle_det_coco1/val.json --images_dir coco-2017/train/images --epochs 10 --batch_size 4 --pretrained
```

### 3. 训练参数

- `--epochs`：训练轮数，默认10
- `--batch_size`：批次大小，默认4
- `--lr`：学习率，默认5e-4
- `--pretrained`：使用COCO预训练权重
- `--min_size`：图像最小尺寸，默认800
- `--max_size`：图像最大尺寸，默认1333

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

### Q: 训练时出现显存不足错误
A: 减小batch_size和图像尺寸：`--batch_size 1 --min_size 640 --max_size 1000`

## 输出文件说明

- `overlay.mp4`：带有检测结果的视频
- `events.json`：违停事件详情
- `events.csv`：违停事件表格
- `run_meta.json`：运行配置信息
- `snapshots/`：违停事件截图

## 技术创新点

1. **集成化解决方案**：将ROI标注、目标检测、跟踪、违停判定和结果可视化集成在一个系统中
2. **基于中心点的ROI判定**：使用车辆bounding box中心点进行ROI判定，提高准确性
3. **多目标跟踪**：实现了基于IoU的多目标跟踪，支持短时丢检容忍
4. **音频保留**：使用MoviePy保留原始视频的音频，提高结果视频的完整性
5. **用户友好的标注工具**：提供直观的多边形标注界面，支持撤销和取消操作

## 应用场景

- **停车场管理**：自动检测停车场内的违停车辆
- **路边停车监控**：监控路边停车位的使用情况
- **交通执法辅助**：为交通执法提供违停证据
- **智能交通系统**：作为智能交通系统的一部分，提高交通管理效率

## 系统要求

- Python 3.8+
- PyTorch 1.8+
- OpenCV
- Pillow
- Tkinter
- MoviePy (可选，用于音频处理)

## 性能优化

1. **GPU加速**：使用CUDA进行模型推理，提高检测速度
2. **批量推理**：一次处理多个图像，提高推理效率
3. **模型量化**：将模型参数从浮点数转换为整数，减少模型大小和推理时间
4. **多线程处理**：使用子进程运行标注和检测，避免UI阻塞

## 未来扩展

1. **实时检测**：支持实时视频流的违停检测
2. **多类别检测**：扩展到其他类型的车辆和物体检测
3. **远程监控**：支持远程查看检测结果和实时报警
4. **模型优化**：使用更轻量级的模型，如YOLO系列，提高实时性能
5. **数据增强**：增加更多数据增强策略，提高模型泛化能力

## 项目预览：
![990a753f4edf926598cc56e6086d7ea4](https://github.com/user-attachments/assets/79a6d900-f65b-4e45-8c66-42a4fbee7f7a)
<img width="1002" height="630" alt="7ae521b09ebfccae9d95b51e09dcb0ec" src="https://github.com/user-attachments/assets/8d941f7b-2d49-4f33-959e-cfe422af54f9" />

##效果展示：
###场景一：
<img width="2153" height="1117" alt="image" src="https://github.com/user-attachments/assets/59b409a1-a476-4248-8655-121713ffd164" />
<img width="2156" height="1120" alt="image" src="https://github.com/user-attachments/assets/e4673591-9508-468a-a420-0abff5f4ee16" />
###场景二：
<img width="2159" height="1125" alt="image" src="https://github.com/user-attachments/assets/9c154f5e-afd9-4664-a6e6-d7d704f42470" />
<img width="2159" height="1119" alt="image" src="https://github.com/user-attachments/assets/34032fe1-482b-488d-9a57-baa5a8a872a5" />





