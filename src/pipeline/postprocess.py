# Image to SVG/DXF

import cv2
import numpy as np
import svgwrite
import os

def image_to_svg(image_path, output_path="data/gen/generated.svg", scale=1.0):
    # Load image and preprocess
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, threshold1=50, threshold2=150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create SVG canvas
    height, width = edges.shape
    dwg = svgwrite.Drawing(output_path, size=(width, height))

    for contour in contours:
        points = [(int(p[0][0] * scale), int(p[0][1] * scale)) for p in contour]
        if len(points) > 1:
            dwg.add(dwg.polyline(points, stroke='black', fill='none', stroke_width=1))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dwg.save()
    return output_path
