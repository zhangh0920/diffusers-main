export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="/home/zl/zh/code/diffusers-main/dog"
export CLASS_DIR="/home/zl/zh/data/dreambooth_dataset/dog8"
export OUTPUT_DIR="/home/zl/zh/code/diffusers-main/examples/dreambooth/output_model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800 \
#  --push_to_hub