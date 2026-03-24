"""
使用空白图像进行推理的示例脚本
基于原始inference.py，集成了空白图像生成功能
"""

import os
import sys
import argparse

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PIL import Image
import torch
import transformers
from diffusers import FlowMatchEulerDiscreteScheduler

from models.adapter_models import *
from utils.sd3_utils import *
from utils.utils import save_image, post_process
from utils.data_processor import UserInputProcessor
from tools.generate_blank_images import generate_blank_images


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="使用空白图像进行PosterMaker推理（纯文字海报生成）"
    )
    
    # 模型路径
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--controlnet_model_name_or_path", type=str, default=None)
    parser.add_argument("--controlnet_model_name_or_path2", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None, required=False)
    
    # 图像尺寸
    parser.add_argument("--resolution_h", type=int, default=1024)
    parser.add_argument("--resolution_w", type=int, default=1024)
    
    # ControlNet层数
    parser.add_argument("--ctrl_layers", type=int, default=23)
    
    # 推理参数
    parser.add_argument('--num_inference_steps', type=int, default=28)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_float16", action='store_true')
    
    # 空白图像生成参数
    parser.add_argument("--generate_blank", action='store_true',
                       help="自动生成空白图像和遮罩")
    parser.add_argument("--filename", type=str, default='blank_poster_demo',
                       help="文件名（不含扩展名）")
    parser.add_argument("--mask_type", type=str, default='full',
                       choices=['full', 'center'],
                       help="遮罩类型：full(全黑) 或 center(中心保留)")
    parser.add_argument("--bg_color", type=int, nargs=3, default=[255, 255, 255],
                       help="背景颜色 RGB")
    
    # 输出目录
    parser.add_argument("--output_dir", type=str, default='./images/results')
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    return args


def main():
    args = parse_args()
    
    print("=" * 80)
    print("PosterMaker - 纯文字海报生成（使用空白图像）")
    print("=" * 80)
    
    # 如果需要生成空白图像
    if args.generate_blank:
        print("\n步骤 1/4: 生成空白图像和遮罩...")
        print("-" * 80)
        rgba_path, mask_path = generate_blank_images(
            output_dir_rgba='./images/rgba_images',
            output_dir_mask='./images/subject_masks',
            filename=args.filename,
            width=args.resolution_w,
            height=args.resolution_h,
            bg_color=tuple(args.bg_color),
            mask_type=args.mask_type
        )
    else:
        # 使用现有图像
        rgba_path = f'./images/rgba_images/{args.filename}.png'
        mask_path = f'./images/subject_masks/{args.filename}.png'
        
        if not os.path.exists(rgba_path) or not os.path.exists(mask_path):
            print(f"\n错误: 找不到图像文件")
            print(f"  RGBA: {rgba_path}")
            print(f"  Mask: {mask_path}")
            print(f"\n提示: 使用 --generate_blank 参数自动生成空白图像")
            return
    
    # 加载模型
    print("\n步骤 2/4: 加载模型...")
    print("-" * 80)
    
    # 加载text encoders
    print("加载 Text Encoders...")
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        args, text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
    )
    
    # 加载tokenizers
    print("加载 Tokenizers...")
    tokenizer_one = transformers.CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision,
    )
    tokenizer_two = transformers.CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision,
    )
    tokenizer_three = transformers.T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_3", revision=args.revision,
    )
    
    # 加载VAE
    print("加载 VAE...")
    vae = load_vae(args)
    
    # 加载SD3
    print("加载 SD3 Transformer...")
    transformer = load_transfomer(args)
    
    # 加载scheduler
    print("加载 Scheduler...")
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    
    # 创建ControlNets
    print("加载 SceneGenNet...")
    controlnet_inpaint = load_controlnet(
        args, transformer, additional_in_channel=1, 
        num_layers=args.ctrl_layers, scratch=True
    )
    
    print("加载 TextRenderNet...")
    controlnet_text = load_controlnet(
        args, transformer, additional_in_channel=0, scratch=True
    )
    
    # 加载adapter
    print("加载 Adapter...")
    adapter = LinearAdapterWithLayerNorm(128, 4096)
    
    # 加载权重
    print("加载模型权重...")
    controlnet_inpaint.load_state_dict(
        torch.load(args.controlnet_model_name_or_path, map_location='cpu')
    )
    textrender_net_state_dict = torch.load(
        args.controlnet_model_name_or_path2, map_location='cpu'
    )
    controlnet_text.load_state_dict(textrender_net_state_dict['controlnet_text'])
    adapter.load_state_dict(textrender_net_state_dict['adapter'])
    
    # 设置device和dtype
    weight_dtype = torch.float16 if args.use_float16 else torch.float32
    device = torch.device("cuda")
    
    # 移动模型到device
    print("移动模型到GPU...")
    vae.to(device=device)
    text_encoder_one.to(device=device, dtype=weight_dtype)
    text_encoder_two.to(device=device, dtype=weight_dtype)
    text_encoder_three.to(device=device, dtype=weight_dtype)
    controlnet_inpaint.to(device=device, dtype=weight_dtype)
    controlnet_text.to(device=device, dtype=weight_dtype)
    adapter.to(device=device, dtype=weight_dtype)
    
    # 创建pipeline
    print("创建 Pipeline...")
    from pipelines.pipeline_sd3 import StableDiffusion3ControlNetPipeline
    pipeline = StableDiffusion3ControlNetPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_config(noise_scheduler.config),
        vae=vae,
        transformer=transformer,
        text_encoder=text_encoder_one,
        tokenizer=tokenizer_one,
        text_encoder_2=text_encoder_two,
        tokenizer_2=tokenizer_two,
        text_encoder_3=text_encoder_three,
        tokenizer_3=tokenizer_three,
        controlnet_inpaint=controlnet_inpaint,
        controlnet_text=controlnet_text,
        adapter=adapter,
    )
    pipeline = pipeline.to(dtype=weight_dtype, device=device)
    
    print("✓ 模型加载完成")
    
    # 准备输入数据
    print("\n步骤 3/4: 准备输入数据...")
    print("-" * 80)
    
    # 用户输入示例
    prompt = """A traditional Chinese landscape painting style featuring misty mountains, flowing rivers, ancient pine trees, and ethereal clouds, rendered in ink wash painting techniques, creating a serene and poetic atmosphere."""
    
    texts = [
        {"content": "大", "pos": [30, 100, 90, 174]},
        {"content": "漠", "pos": [30, 174, 90, 248]},
        {"content": "孤", "pos": [30, 248, 90, 322]},
        {"content": "烟", "pos": [30, 322, 90, 396]},
        {"content": "直", "pos": [30, 396, 90, 470]}
    ]
    
    print(f"Prompt: {prompt[:100]}...")
    print(f"文字数量: {len(texts)}")
    for i, text in enumerate(texts):
        print(f"  文字{i+1}: '{text['content']}' at {text['pos']}")
    
    # 加载图像
    image = cv2.imread(rgba_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # 预处理
    data_processor = UserInputProcessor()
    input_data = data_processor(
        image=image,
        mask=mask,
        texts=texts,
        prompt=prompt
    )
    
    print("✓ 数据预处理完成")
    
    # 推理
    print("\n步骤 4/4: 生成图像...")
    print("-" * 80)
    print(f"推理步数: {args.num_inference_steps}")
    print(f"CFG Scale: {args.cfg_scale}")
    print(f"生成数量: {args.num_images_per_prompt}")
    print(f"随机种子: {args.seed}")
    
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    results = pipeline(
        prompt=input_data['prompt'],
        negative_prompt='deformed, distorted, disfigured, poorly drawn, bad anatomy, '
                       'wrong anatomy, extra limb, missing limb, floating limbs, '
                       'mutated hands and fingers, disconnected limbs, mutation, '
                       'mutated, ugly, disgusting, blurry, amputation, NSFW',
        height=args.resolution_h,
        width=args.resolution_w,
        control_image=[input_data['cond_image_inpaint'], input_data['controlnet_im']],
        control_mask=input_data['control_mask'],
        text_embeds=input_data['text_embeds'],
        num_inference_steps=args.num_inference_steps,
        generator=generator,
        controlnet_conditioning_scale=1.0,
        guidance_scale=args.cfg_scale,
        num_images_per_prompt=args.num_images_per_prompt,
    ).images
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    if len(results) == 1:
        image = results[0]
        image = post_process(image, input_data['target_size'])
        output_path = os.path.join(args.output_dir, f"{args.filename}.jpg")
        save_image(image, output_path)
        print(f"\n✓ 图像已保存: {output_path}")
    else:
        for i, image in enumerate(results):
            image = post_process(image, input_data['target_size'])
            output_path = os.path.join(args.output_dir, f"{args.filename}_{i}.jpg")
            save_image(image, output_path)
            print(f"\n✓ 图像 {i+1}/{len(results)} 已保存: {output_path}")
    
    print("\n" + "=" * 80)
    print("生成完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
