# Main script

from src.pipeline.generator import generate_image
from src.pipeline.postprocess import image_to_svg

prompt = "Flat L-shaped bracket with three holes"
img_path = generate_image(prompt)
svg_path = image_to_svg(img_path)

print(f"SVG saved at: {svg_path}")


