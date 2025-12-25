"""
批量推理脚本：从proj生成GT图像
用法:
    python inference_full_dataset.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --data_roots /path/to/data1 /path/to/data2 \
        --output_root ./inference_results \
        --num_steps 50
"""

import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Proj to GT Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_roots', type=str, nargs='+', required=True, help='Data root directories')
    parser.add_argument('--output_root', type=str, default=None, help='Output directory')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512], help='Image size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    return parser.parse_args()


def main():
    args = parse_args()

    # 导入模块
    from src.lightning_depth import LidarDiffusionModule

    # 设置输出目录
    if args.output_root is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_root = f"./inference_results_{timestamp}"

    os.makedirs(args.output_root, exist_ok=True)
    print(f"输出目录: {args.output_root}")

    # 加载模型
    print(f"加载模型: {args.checkpoint}")
    model = LidarDiffusionModule.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.to(args.device)
    print("模型加载完成")

    # 图像变换
    transform = transforms.Compose([
        transforms.Resize(tuple(args.image_size)),
        transforms.ToTensor(),
    ])

    # 定义需要处理的图像名称
    image_names = ['FL.jpg', 'FN.jpg', 'FR.jpg', 'FW.jpg', 'RL.jpg', 'RN.jpg', 'RR.jpg']

    # 统计
    total_processed = 0
    total_errors = 0

    # 遍历所有数据根目录
    for data_root in args.data_roots:
        print(f"\n处理数据目录: {data_root}")

        if not os.path.exists(data_root):
            print(f"警告: 路径不存在 {data_root}")
            continue

        # 遍历所有时间戳目录
        timestamp_dirs = sorted([d for d in os.listdir(data_root)
                                  if os.path.isdir(os.path.join(data_root, d))])

        for timestamp_dir in tqdm(timestamp_dirs, desc=f"处理 {os.path.basename(data_root)}"):
            timestamp_path = os.path.join(data_root, timestamp_dir)

            # 支持大小写 gt/GT
            gt_dir = os.path.join(timestamp_path, 'gt')
            if not os.path.exists(gt_dir):
                gt_dir = os.path.join(timestamp_path, 'GT')

            proj_dir = os.path.join(timestamp_path, 'proj')

            if not (os.path.exists(gt_dir) and os.path.exists(proj_dir)):
                continue

            # 创建输出目录
            output_timestamp_dir = os.path.join(args.output_root, timestamp_dir)
            os.makedirs(f"{output_timestamp_dir}/proj", exist_ok=True)
            os.makedirs(f"{output_timestamp_dir}/GT", exist_ok=True)
            os.makedirs(f"{output_timestamp_dir}/generated", exist_ok=True)
            os.makedirs(f"{output_timestamp_dir}/comparison", exist_ok=True)

            # 处理每个视角
            for img_name in image_names:
                proj_path = os.path.join(proj_dir, img_name)
                gt_path = os.path.join(gt_dir, img_name)

                if not (os.path.exists(proj_path) and os.path.exists(gt_path)):
                    continue

                try:
                    # 加载图像
                    proj_img = Image.open(proj_path).convert('RGB')
                    gt_img = Image.open(gt_path).convert('RGB')

                    proj_tensor = transform(proj_img).unsqueeze(0).to(args.device)
                    gt_tensor = transform(gt_img).unsqueeze(0).to(args.device)

                    # 生成图像
                    with torch.no_grad():
                        generated = model.generate_from_lidar(proj_tensor, num_inference_steps=args.num_steps)
                        generated = torch.clamp(generated, 0, 1)

                    # 保存结果
                    view_name = img_name.split('.')[0]

                    # 保存原始proj
                    save_image(proj_tensor, f"{output_timestamp_dir}/proj/{img_name}")

                    # 保存原始GT
                    save_image(gt_tensor, f"{output_timestamp_dir}/GT/{img_name}")

                    # 保存生成结果
                    save_image(generated, f"{output_timestamp_dir}/generated/{img_name}")

                    # 保存对比图 (proj | generated | GT)
                    comparison = torch.cat([proj_tensor, generated, gt_tensor], dim=3)
                    save_image(comparison, f"{output_timestamp_dir}/comparison/{view_name}_comparison.jpg")

                    total_processed += 1

                except Exception as e:
                    print(f"处理失败 {proj_path}: {e}")
                    total_errors += 1

    print(f"\n=== 推理完成 ===")
    print(f"成功处理: {total_processed} 张图像")
    print(f"处理失败: {total_errors} 张图像")
    print(f"输出目录: {args.output_root}")


if __name__ == "__main__":
    main()
