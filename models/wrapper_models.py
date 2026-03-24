# import random
# import torch
# from torch import nn

# class WrapperModel_SD3_ControlNet_with_Adapter(nn.Module):
#     def __init__(self, controlnet, adapter, **kwargs):
#         super(WrapperModel_SD3_ControlNet_with_Adapter, self).__init__()
#         self.controlnet = controlnet
#         self.adapter = adapter

#     def forward(self, noisy_model_input, timestep, prompt_embeds, controlnet_pooled_projections, controlnet_cond, text_embeds, **kwargs):
#         # text embed shape: [b, 128, 1472]
#         text_features = self.adapter(text_embeds) # [b, 128, 4096]
#         # controlnet
#         control_block_samples = self.controlnet(
#             hidden_states=noisy_model_input,
#             timestep=timestep,
#             encoder_hidden_states=text_features,
#             pooled_projections=controlnet_pooled_projections,
#             controlnet_cond=controlnet_cond,
#             return_dict=False,
#         )[0]
#         return control_block_samples


import torch
from torch import nn

class WrapperModel_SD3_ControlNet_with_Adapter(nn.Module):
    def __init__(self, controlnet, adapter, use_style_cond=False, style_in_dim=2560, text_embed_dim=128):
        """
        :param use_style_cond: 结构开关。True时初始化风格投影网络。
        :param style_in_dim: OCR特征展平后的维度。默认 PPOCRv3 Neck输出可能是 40*64=2560
        :param text_embed_dim: 你的 text_embeds 的维度，默认是 128
        """
        super(WrapperModel_SD3_ControlNet_with_Adapter, self).__init__()
        self.controlnet = controlnet
        self.adapter = adapter
        
        # --- 新增：消融实验与结构开关 ---
        self.use_style_cond = use_style_cond
        
        if self.use_style_cond:
            # 风格投影层：将 OCR 高维序列特征压缩到与 text_embeds 相同的 128 维
            self.style_proj = nn.Sequential(
                nn.Linear(style_in_dim, 512),
                nn.SiLU(),
                nn.Linear(512, text_embed_dim),
                nn.LayerNorm(text_embed_dim)
            )
        else:
            self.style_proj = None
        # --------------------------------

    def forward(
        self, 
        noisy_model_input, 
        timestep, 
        prompt_embeds, 
        controlnet_pooled_projections, 
        controlnet_cond, 
        text_embeds, 
        style_features=None,  # 新增：从外部传入的 OCR 提取特征
        enable_style=True,    # 新增：动态消融开关，默认为 True
        **kwargs
    ):
        # ===== 新增：风格特征注入模块 (方法 A: 特征相加) =====
        if self.use_style_cond and enable_style and (style_features is not None) and (self.style_proj is not None):
            # 假设传入的 style_features 形状为 [Batch, TimeSteps, Channels] (例如 [B, 40, 64])
            batch_size = style_features.shape[0]
            
            # 1. 展平序列特征 -> [B, T * C]
            # style_features_flat = style_features.view(batch_size, -1) 
            style_features_flat = style_features.reshape(batch_size, -1)
            
            # 2. 通过投影层降维 -> [B, 128]
            style_embeds = self.style_proj(style_features_flat)
            
            # 3. 增加维度以匹配 text_embeds -> [B, 1, 128]
            style_embeds = style_embeds.unsqueeze(1)
            
            # 4. 特征相加 (利用广播机制，加到每一个字的内容特征上)
            text_embeds = text_embeds + style_embeds
        # =====================================================

        # 原有逻辑：送入 Adapter (此时 text_embeds 已经包含了风格信息)
        text_features = self.adapter(text_embeds) # [b, 128, 4096]
        
        # controlnet 前向传播
        control_block_samples = self.controlnet(
            hidden_states=noisy_model_input,
            timestep=timestep,
            encoder_hidden_states=text_features,
            pooled_projections=controlnet_pooled_projections,
            controlnet_cond=controlnet_cond,
            return_dict=False,
        )[0]
        
        return control_block_samples