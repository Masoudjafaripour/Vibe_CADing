# Text -> CAD generation

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import os

MODEL_ID = "runwayml/stable-diffusion-v1-5"
CONTROLNET_ID = "lllyasviel/control_v11p_sd15_canny"  # good for edge/CAD-like generation

def load_pipeline():
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        MODEL_ID, controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # pipe.enable_xformers_memory_efficient_attention()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print("⚠️ xformers not available – continuing without it.")

    pipe.to("cuda")
    return pipe

def generate_image(prompt, save_path="data/gen/generated.png"):
    pipe = load_pipeline()

    # You can add real edge maps here later; for now use a blank or simple guide
    canny_image = Image.new("RGB", (512, 512), (255, 255, 255))

    result = pipe(prompt, image=canny_image, num_inference_steps=20)
    image = result.images[0]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)
    return save_path
