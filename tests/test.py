import torch
import torchvision
from torchvision.ops import nms

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Torchvision NMS loaded successfully.")

import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("GPU count:", torch.cuda.device_count())