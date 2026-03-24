# PosterMaker 工具集

## 空白图像生成工具 (generate_blank_images.py)

该工具用于生成空白的主体图像和遮罩，以便在没有产品主体的情况下使用PosterMaker生成纯文字海报。

### 功能说明

PosterMaker原本设计用于产品海报生成，需要产品主体图像和对应的遮罩。此工具提供了一种替代方案：
- 生成空白的RGBA图像作为"主体"
- 生成全黑或部分遮罩，让模型在指定区域生成背景

### 使用方法

#### 1. 生成单张空白图像

```bash
cd /home/610-zzy/CalliEdit-20251226-Version1.0/PosterMaker-main

# 基础用法 - 生成默认配置的图像
python tools/generate_blank_images.py

# 自定义文件名
python tools/generate_blank_images.py --filename my_blank_poster

# 自定义尺寸
python tools/generate_blank_images.py --width 1024 --height 768

# 自定义背景颜色（白色）
python tools/generate_blank_images.py --bg-color 255 255 255

# 使用中心保留遮罩（模拟有主体的情况）
python tools/generate_blank_images.py --mask-type center
```

#### 2. 批量生成多张图像

```bash
# 批量生成5张
python tools/generate_blank_images.py --batch 5

# 批量生成10张，自定义前缀
python tools/generate_blank_images.py --batch 10 --prefix test_blank

# 批量生成并指定输出目录
python tools/generate_blank_images.py --batch 3 \
    --output-rgba ./my_images/rgba \
    --output-mask ./my_images/masks
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--filename` | str | `blank_poster` | 输出文件名（不含扩展名） |
| `--width` | int | `1024` | 图像宽度 |
| `--height` | int | `1024` | 图像高度 |
| `--bg-color` | int×3 | `255 255 255` | 背景颜色(R G B) |
| `--mask-type` | str | `full` | 遮罩类型：`full`或`center` |
| `--batch` | int | `None` | 批量生成数量 |
| `--prefix` | str | `blank` | 批量生成文件名前缀 |
| `--output-rgba` | str | `./images/rgba_images` | RGBA图像输出目录 |
| `--output-mask` | str | `./images/subject_masks` | 遮罩图像输出目录 |

### 遮罩类型说明

#### `full` - 全黑遮罩（推荐）
- 整个图像区域为黑色（值为0）
- 表示整个区域都需要模型生成背景
- 适合生成纯文字海报

#### `center` - 中心保留遮罩
- 中心40%区域为白色（值为255），周围为黑色
- 模拟有主体物品的情况
- 中心区域会被保留，周围生成背景

### 使用示例

#### 示例1：生成纯文字海报

```bash
# 1. 生成空白图像
python tools/generate_blank_images.py --filename text_poster_001

# 2. 修改 inference.py 中的配置
filename = 'text_poster_001'
image_path = f'./images/rgba_images/{filename}.png'
mask_path  = f'./images/subject_masks/{filename}.png'
prompt = "A modern minimalist background with soft gradient colors"
texts = [
    {"content": "新品上市", "pos": [200, 300, 824, 400]},
    {"content": "限时优惠", "pos": [300, 450, 724, 550]}
]

# 3. 运行推理
python inference.py \
    --pretrained_model_name_or_path='./checkpoints/stable-diffusion-3-medium-diffusers/' \
    --controlnet_model_name_or_path='./checkpoints/our_weights/scenegen_net-0415.pth' \
    --controlnet_model_name_or_path2='./checkpoints/our_weights/textrender_net-0415.pth' \
    --seed=42 \
    --num_images_per_prompt=4
```

#### 示例2：批量生成测试图像

```bash
# 批量生成10张测试图像
python tools/generate_blank_images.py --batch 10 --prefix test

# 生成的文件：
# - ./images/rgba_images/test_000.png ~ test_009.png
# - ./images/subject_masks/test_000.png ~ test_009.png
```

### 注意事项

1. **生成质量**：由于这不是PosterMaker的设计初衷，生成的纯文字海报质量可能不如有真实产品主体的海报

2. **文字布局**：建议合理规划文字位置，避免过于集中或超出图像边界

3. **背景描述**：prompt应该描述想要的背景风格，但避免提及文字内容

4. **目录结构**：生成的图像会自动保存到PosterMaker期望的目录结构中

### 输出结果

运行脚本后，会在指定目录生成：
- `{filename}.png` - RGBA格式的空白主体图像（保存在rgba_images目录）
- `{filename}.png` - 对应的遮罩图像（保存在subject_masks目录）

这些图像可以直接用于PosterMaker的推理流程。

### 故障排除

**问题：生成的海报没有文字**
- 检查texts参数是否正确设置
- 确保文字位置在图像范围内

**问题：背景生成效果不理想**
- 尝试不同的prompt描述
- 调整seed参数多次生成
- 考虑使用center遮罩类型

**问题：文件保存失败**
- 确保输出目录存在或有创建权限
- 检查磁盘空间是否充足
