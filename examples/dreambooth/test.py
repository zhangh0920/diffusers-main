from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch

# # Load the pipeline with the same arguments (model, revision) that were used for training
# model_id = "CompVis/stable-diffusion-v1-4"
#
# unet = UNet2DConditionModel.from_pretrained("/home/zl/zh/code/diffusers-main/examples/dreambooth/output_model/unet")
#
# # if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
# # text_encoder = CLIPTextModel.from_pretrained("/sddata/dreambooth/daruma-v2-1/checkpoint-100/text_encoder")
#
# pipeline = DiffusionPipeline.from_pretrained(
#     model_id, unet=unet, dtype=torch.float16, use_safetensors=True
# )
# pipeline.to("cuda")
#
# # Perform inference, or save, or push to the hub
# # pipeline.save_pretrained("dreambooth-pipeline")
#
#
# prompt = "A photo of sks dog in a bucket"
# image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
#
# image.save("dog-bucket.png")




import torch
from diffusers import StableDiffusionAttendAndExcitePipeline

pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda")

prompt = "a cat and a frog"
pipe.get_indices(prompt)

token_indices = [2, 5]
seed = 6141
generator = torch.Generator("cuda").manual_seed(seed)
images = pipe(prompt=prompt, token_indices=token_indices, guidance_scale=7.5, generator=generator, num_inference_steps=50, max_iter_to_alter=25,).images

image = images[0]
image.save(f"../images/{prompt}_{seed}.png")