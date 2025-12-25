import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel


class LidarDiffusionModel(nn.Module):
    """
    基于Stable Diffusion UNet的扩散模型，使用proj图像作为条件生成GT图像。
    """
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", lidar_channels=3):
        super().__init__()

        # 加载预训练的UNet
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

        # 原始UNet输入通道是4（latent），我们需要增加lidar条件通道
        original_in_channels = self.unet.config.in_channels  # 4
        new_in_channels = original_in_channels + lidar_channels  # 4 + 3 = 7

        # 修改第一个卷积层以接受额外的通道
        original_conv = self.unet.conv_in
        self.unet.conv_in = nn.Conv2d(
            new_in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding
        )

        # 初始化新的卷积层
        with torch.no_grad():
            # 复制原始权重到前4个通道
            self.unet.conv_in.weight[:, :original_in_channels] = original_conv.weight
            # 新通道用小随机值初始化
            self.unet.conv_in.weight[:, original_in_channels:] = torch.randn_like(
                self.unet.conv_in.weight[:, original_in_channels:]
            ) * 0.01
            # 复制偏置
            if original_conv.bias is not None:
                self.unet.conv_in.bias = nn.Parameter(original_conv.bias.clone())

        # 创建一个空的encoder_hidden_states（因为我们不使用文本条件）
        self.register_buffer(
            'null_text_embedding',
            torch.zeros(1, 77, self.unet.config.cross_attention_dim)
        )

    def forward(self, noisy_latents, lidar_cond, timesteps):
        """
        前向传播
        Args:
            noisy_latents: 加噪的latent [B, 4, H, W]
            lidar_cond: LiDAR/proj条件图像 [B, 3, H, W]
            timesteps: 时间步 [B]
        Returns:
            noise_pred: 预测的噪声 [B, 4, H, W]
        """
        # 拼接latent和lidar条件
        concat_input = torch.cat([noisy_latents, lidar_cond], dim=1)  # [B, 7, H, W]

        # 准备encoder_hidden_states（空文本嵌入）
        batch_size = noisy_latents.size(0)
        encoder_hidden_states = self.null_text_embedding.expand(batch_size, -1, -1)

        # UNet前向传播
        noise_pred = self.unet(
            concat_input,
            timesteps,
            encoder_hidden_states=encoder_hidden_states
        ).sample

        return noise_pred
