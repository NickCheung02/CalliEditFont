# python inference.py \
#  --pretrained_model_name_or_path='./checkpoints/stable-diffusion-3-medium-diffusers/' \
#  --controlnet_model_name_or_path='./checkpoints/our_weights/scenegen_net-0415.pth' \
#  --controlnet_model_name_or_path2='./checkpoints/our_weights/textrender_net-0415.pth' \
#  --seed=42 \
#  --num_images_per_prompt=4 \
#  --use_float16


# python inference.py \
#  --pretrained_model_name_or_path='./checkpoints/stable-diffusion-3-medium-diffusers/' \
#  --controlnet_model_name_or_path='./checkpoints/our_weights/scenegen_net-1m-0415.pth' \
#  --controlnet_model_name_or_path2='./checkpoints/our_weights/textrender_net-1m-0415.pth' \
#  --seed=42 \
#  --num_images_per_prompt=4 \
#  --use_float16

 python inference.py \
 --pretrained_model_name_or_path='./checkpoints/stable-diffusion-3-medium-diffusers/' \
 --controlnet_model_name_or_path2='./checkpoints_training/postermaker_stage1_20260322_203813/4000_net_postermaker_stage1_20260322_203813.pth' \
 --controlnet_model_name_or_path='./checkpoints_training/postermaker_stage2_20260323_212816/8000_net_postermaker_stage2_20260323_212816.pth' \
 --seed=42 \
 --num_images_per_prompt=4 \
 --use_float16