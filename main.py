import torch
from torch import autocast
from authtoken import auth_token
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = torch.device('cpu')

pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=auth_token)
pipe = pipe.to(device)

prompt = input("Write Please!: ")
with autocast("cuda"):
    image = pipe(prompt, guidance_scale = 7.5).images[0]

image.save("test_image.png")