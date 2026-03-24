"""
生成空白主体图像和遮罩的工具
用于在没有产品主体的情况下使用PosterMaker生成纯文字海报
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path


def generate_blank_images(
    output_dir_rgba='./images/rgba_images',
    output_dir_mask='./images/subject_masks',
    filename='blank_poster',
    width=1024,
    height=1024,
    bg_color=(255, 255, 255),  # 白色背景
    mask_type='full'  # 'full': 全黑遮罩(全部生成), 'center': 中心区域遮罩
):
    """
    生成空白主体图像和对应的遮罩
    
    Args:
        output_dir_rgba: RGBA图像保存目录
        output_dir_mask: 遮罩图像保存目录
        filename: 文件名（不含扩展名）
        width: 图像宽度
        height: 图像高度
        bg_color: 背景颜色 (R, G, B)
        mask_type: 遮罩类型
            - 'full': 全黑遮罩，整个区域都会生成背景
            - 'center': 中心保留白色区域（模拟有主体的情况）
    """
    
    # 创建输出目录
    os.makedirs(output_dir_rgba, exist_ok=True)
    os.makedirs(output_dir_mask, exist_ok=True)
    
    # 1. 创建RGBA图像（空白背景）
    rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
    rgba_image[:, :, 0] = bg_color[2]  # B
    rgba_image[:, :, 1] = bg_color[1]  # G
    rgba_image[:, :, 2] = bg_color[0]  # R
    rgba_image[:, :, 3] = 255  # Alpha通道全不透明
    
    # 保存RGBA图像
    rgba_path = os.path.join(output_dir_rgba, f'{filename}.png')
    cv2.imwrite(rgba_path, rgba_image)
    print(f"✓ 已生成RGBA图像: {rgba_path}")
    
    # 2. 创建遮罩图像
    if mask_type == 'full':
        # 全黑遮罩 - 整个区域都需要生成背景
        mask = np.zeros((height, width), dtype=np.uint8)
        print("  遮罩类型: 全黑遮罩（整个区域生成背景）")
    
    elif mask_type == 'center':
        # 中心区域为白色（保留），周围为黑色（生成）
        mask = np.zeros((height, width), dtype=np.uint8)
        center_h, center_w = int(height * 0.4), int(width * 0.4)
        top = (height - center_h) // 2
        left = (width - center_w) // 2
        mask[top:top+center_h, left:left+center_w] = 255
        print(f"  遮罩类型: 中心保留区域 ({center_w}x{center_h})")
    
    else:
        raise ValueError(f"不支持的遮罩类型: {mask_type}")
    
    # 保存遮罩图像
    mask_path = os.path.join(output_dir_mask, f'{filename}.png')
    cv2.imwrite(mask_path, mask)
    print(f"✓ 已生成遮罩图像: {mask_path}")
    
    return rgba_path, mask_path


def generate_batch_images(
    count=5,
    output_dir_rgba='./images/rgba_images',
    output_dir_mask='./images/subject_masks',
    prefix='blank',
    **kwargs
):
    """
    批量生成多个空白图像和遮罩
    
    Args:
        count: 生成数量
        output_dir_rgba: RGBA图像保存目录
        output_dir_mask: 遮罩图像保存目录
        prefix: 文件名前缀
        **kwargs: 传递给generate_blank_images的其他参数
    """
    print(f"\n开始批量生成 {count} 组图像...\n")
    
    results = []
    for i in range(count):
        filename = f"{prefix}_{i:03d}"
        print(f"[{i+1}/{count}] 生成: {filename}")
        rgba_path, mask_path = generate_blank_images(
            output_dir_rgba=output_dir_rgba,
            output_dir_mask=output_dir_mask,
            filename=filename,
            **kwargs
        )
        results.append((rgba_path, mask_path))
        print()
    
    print(f"✓ 批量生成完成！共生成 {count} 组图像")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="生成空白主体图像和遮罩，用于PosterMaker纯文字海报生成"
    )
    parser.add_argument(
        '--filename', 
        type=str, 
        default='blank_poster',
        help='输出文件名（不含扩展名）'
    )
    parser.add_argument(
        '--width', 
        type=int, 
        default=1024,
        help='图像宽度'
    )
    parser.add_argument(
        '--height', 
        type=int, 
        default=1024,
        help='图像高度'
    )
    parser.add_argument(
        '--bg-color', 
        type=int, 
        nargs=3,
        default=[255, 255, 255],
        help='背景颜色 RGB，例如: --bg-color 255 255 255'
    )
    parser.add_argument(
        '--mask-type',
        type=str,
        choices=['full', 'center'],
        default='full',
        help='遮罩类型: full(全黑，全部生成) 或 center(中心保留)'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=None,
        help='批量生成数量，不指定则生成单张'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='blank',
        help='批量生成时的文件名前缀'
    )
    parser.add_argument(
        '--output-rgba',
        type=str,
        default='./images/rgba_images',
        help='RGBA图像输出目录'
    )
    parser.add_argument(
        '--output-mask',
        type=str,
        default='./images/subject_masks',
        help='遮罩图像输出目录'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PosterMaker 空白图像生成工具")
    print("=" * 60)
    print(f"图像尺寸: {args.width}x{args.height}")
    print(f"背景颜色: RGB{tuple(args.bg_color)}")
    print(f"遮罩类型: {args.mask_type}")
    print("=" * 60)
    print()
    
    if args.batch:
        # 批量生成
        generate_batch_images(
            count=args.batch,
            output_dir_rgba=args.output_rgba,
            output_dir_mask=args.output_mask,
            prefix=args.prefix,
            width=args.width,
            height=args.height,
            bg_color=tuple(args.bg_color),
            mask_type=args.mask_type
        )
    else:
        # 单张生成
        generate_blank_images(
            output_dir_rgba=args.output_rgba,
            output_dir_mask=args.output_mask,
            filename=args.filename,
            width=args.width,
            height=args.height,
            bg_color=tuple(args.bg_color),
            mask_type=args.mask_type
        )
    
    print("\n" + "=" * 60)
    print("生成完成！现在可以使用这些图像进行推理：")
    print(f"  python inference.py \\")
    print(f"    --pretrained_model_name_or_path='./checkpoints/stable-diffusion-3-medium-diffusers/' \\")
    print(f"    --controlnet_model_name_or_path='./checkpoints/our_weights/scenegen_net-0415.pth' \\")
    print(f"    --controlnet_model_name_or_path2='./checkpoints/our_weights/textrender_net-0415.pth' \\")
    print(f"    --seed=42")
    print("=" * 60)


if __name__ == '__main__':
    main()
