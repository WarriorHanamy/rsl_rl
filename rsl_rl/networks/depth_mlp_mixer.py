import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class MlpBlock(nn.Module):
    """标准的 MLP 块，用于 Token-mixing 或 Channel-mixing"""
    def __init__(self, input_dim, mlp_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, input_dim)
        )

    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):
    """包含 Token-mixing 和 Channel-mixing 的 Mixer 层
        Currently, we disable channel-mixing for depth MLP mixer.
    """
    def __init__(self, num_tokens, tokens_mlp_dim = 64, channels_mlp_dim = 1):
        super().__init__()
        # assuming each sample has been processed normalization.
        # self.ln_token = nn.LayerNorm(hidden_dim)
        self.token_mix = MlpBlock(num_tokens, tokens_mlp_dim)
        
        # self.ln_channel = nn.LayerNorm(hidden_dim)
        # self.channel_mix = MlpBlock(hidden_dim, channels_mlp_dim)

    def forward(self, x):
        # Token-mixing: 作用于空间维度 (S)
        # x shape: [B, S, C]
        # out = self.ln_token(x)
        out = x.transpose(1, 2)  # [B, C, S]
        out = self.token_mix(out)
        out = out.transpose(1, 2)  # [B, S, C]
        x = x + out

        # Channel-mixing: 作用于特征通道维度 (C)
        # out = self.ln_channel(x)
        # x = x + self.channel_mix(out)
        return x

class DepthMLPMixer(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 image_size=(72, 96), 
                 pool_kernel=2, 
                 patch_size=4, 
                 num_blocks=4, 
                 tokens_mlp_dim=256,
                 flatten: bool = True,
                 # channels_mlp_dim now not working. 
                 channels_mlp_dim=512):
        '''
            REC: channels_mlp_dim Don't work currently.
        '''
        super().__init__()

        # Store flatten flag for later use in forward
        self.flatten = flatten

        # 1. First Layer: Max Pooling
        # 减少深度图的分辨率，提取显著特征
        self.max_pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_kernel)
        
        # 计算经过 Max Pool 后的尺寸
        pooled_h = image_size[0] // pool_kernel
        pooled_w = image_size[1] // pool_kernel
        
        # 2. Patch Embedding (通过 Unfold 或 Conv 实现)
        # 将池化后的图像切分为 Patch，并映射到 hidden_dim
        num_patches = (pooled_h // patch_size) * (pooled_w // patch_size)
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'), # 展平为 [Batch, Tokens, Channels]
        )

        # 3. Mixer Blocks
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(num_patches, tokens_mlp_dim, channels_mlp_dim)
            for _ in range(num_blocks)
        ])

        self._num_output_features = num_patches * 1  # 因为通道数固定为1
        # 4. Output Head
        # self.ln_final = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # 输入 x: [B, 1, H, W]
        x = self.max_pool(x)                   # [B, 1, H', W']
        x = self.to_patch_embedding(x)         # [B, S, C]
        
        for block in self.mixer_blocks:
            x = block(x)                       # [B, S, C]
            
        if self.flatten:
            x = x.flatten(1)
        return x


    @property
    def output_dim(self) -> int:
        """Get the total output dimension after flattening."""
        return self._num_output_features

# 示例用法
if __name__ == "__main__":
    # 假设输入 224x224 的深度图
    model = DepthMLPMixer(image_size=(72, 96), pool_kernel=4, patch_size=3)
    dummy_input = torch.randn(1024, 1, 72, 96) # Batch=8
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}") 