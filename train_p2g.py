import os
# 设置要使用的GPU（根据实际情况修改）
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 使用GPU 0和1

import sys
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.utils import save_image
from datetime import datetime
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Proj to GT Diffusion Training')
    parser.add_argument('--config', type=str, default='config_p2g.yaml', help='Path to config file')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to checkpoint for resuming training')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--gpus', type=str, default=None, help='GPU IDs to use (e.g., "0,1" or "2,3")')
    return parser.parse_args()


args = parse_args()

# 如果命令行指定了GPU，则覆盖默认设置
if args.gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

# 检查GPU
print(f"可用GPU数量: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# 导入模块
from src.lightning_depth import LidarDiffusionModule
from dataloader_proj2gt import ProjToGTDataModule

# 加载配置
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

print(f"\n=== 配置信息 ===")
# 支持 data_roots 或 data_root（向后兼容）
data_roots = config['dataset'].get('data_roots', config['dataset'].get('data_root'))
if isinstance(data_roots, list):
    print(f"数据根目录: {len(data_roots)} 个路径")
    for i, path in enumerate(data_roots, 1):
        print(f"  [{i}] {path}")
else:
    print(f"数据根目录: {data_roots}")
print(f"图像尺寸: {config['dataset']['image_size']}")
print(f"批次大小(每GPU): {config['train']['batch_size']}")

# 创建实验目录
if args.exp_name:
    save_dir = f"./experiments/{args.exp_name}"
else:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./experiments/p2g_{timestamp}"

print(f"实验目录: {save_dir}")

os.makedirs(save_dir, exist_ok=True)
os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
os.makedirs(f"{save_dir}/images", exist_ok=True)
os.makedirs(f"{save_dir}/logs", exist_ok=True)

# 数据模块
datamodule = ProjToGTDataModule(
    config=config,
    batch_size=config["train"]["batch_size"],
    num_workers=8,
    val_split=0.2
)

# 模型
model = LidarDiffusionModule(
    lr=float(config["train"].get("lr", 1e-4)),
    t_max=int(config["model"].get("t_max", 1000)),
    weight_decay=float(config["train"].get("weight_decay", 0.0)),
    model_id=config["model"].get("model_id"),
    lidar_channels=config["model"].get("lidar_channels", 3),
)

# 回调函数
best_checkpoint = ModelCheckpoint(
    dirpath=f"{save_dir}/checkpoints",
    filename="best-epoch={epoch:03d}-val_loss={val/loss:.3f}",
    monitor="val/loss",
    mode="min",
    save_top_k=10,
    verbose=True
)

# 定期保存
periodic_checkpoint = ModelCheckpoint(
    dirpath=f"{save_dir}/checkpoints",
    filename="periodic-epoch={epoch:03d}",
    every_n_epochs=20,  # 每20个epoch
    save_top_k=-1,     # 保存所有定期checkpoint
    verbose=True
)

lr_monitor = LearningRateMonitor(logging_interval="step")


# 图像生成回调
class ImageGenerationCallback(Callback):
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % 20 == 0 and trainer.global_rank == 0:
            pl_module.eval()

            epoch_dir = f"{self.save_dir}/images/epoch_{trainer.current_epoch:03d}"
            os.makedirs(epoch_dir, exist_ok=True)

            val_dataloader = trainer.datamodule.val_dataloader()
            val_iterator = iter(val_dataloader)

            for i in range(min(5, len(val_dataloader))):
                try:
                    batch = next(val_iterator)
                    gt = batch["rgb"][0:1].to(pl_module.device)
                    proj = batch["lidar_sparse"][0:1].to(pl_module.device)

                    with torch.no_grad():
                        generated = pl_module.generate_from_lidar(proj, num_inference_steps=20)
                        generated = torch.clamp(generated, 0, 1)

                        # 创建对比图
                        comparison = torch.cat([proj, generated, gt], dim=3)
                        save_image(comparison, f"{epoch_dir}/sample_{i:02d}_comparison.jpg")

                except Exception as e:
                    print(f"生成图像失败: {e}")
                    break


image_callback = ImageGenerationCallback(save_dir)

# TensorBoard日志
logger = TensorBoardLogger(
    save_dir=save_dir,
    name="logs",
    version=None
)

print(f"TensorBoard日志目录: {save_dir}/logs")

# 确定GPU数量
num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    accelerator = "cpu"
    devices = 1
    strategy = "auto"
elif num_gpus == 1:
    accelerator = "gpu"
    devices = 1
    strategy = "auto"
else:
    accelerator = "gpu"
    devices = num_gpus
    strategy = "ddp"

print(f"使用 {devices} 个 {accelerator}")

# 训练器
trainer = pl.Trainer(
    max_epochs=config["train"].get("max_epochs", 100),
    accelerator=accelerator,
    devices=devices,
    strategy=strategy,
    log_every_n_steps=10,
    callbacks=[TQDMProgressBar(), best_checkpoint, periodic_checkpoint, lr_monitor, image_callback],
    logger=logger,
    enable_checkpointing=True,
    gradient_clip_val=1.0,
    check_val_every_n_epoch=1,
    precision="16-mixed",  # 混合精度训练
)

# 开始训练
print("\n=== 开始训练 ===")
if args.ckpt_path:
    print(f"从检查点恢复: {args.ckpt_path}")

trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path)

print("\n=== 训练完成 ===")
print(f"最佳模型: {best_checkpoint.best_model_path}")
