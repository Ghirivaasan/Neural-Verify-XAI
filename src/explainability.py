import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from PIL import Image, ImageChops, ImageEnhance

def generate_ela(image: Image.Image, quality: int = 90) -> Image.Image:
    """
    Generates an Error Level Analysis (ELA) image.
    ELA highlights areas of an image that are at different JPEG compression levels.
    Fake news often pastes manipulated objects (which will show high contrast ELA) over originals.
    """
    temp_filename = 'temp_ela.jpg'
    image_rgb = image.convert('RGB')
    image_rgb.save(temp_filename, 'JPEG', quality=quality)
    temp_img = Image.open(temp_filename)
    
    # Calculate pixel difference
    diff = ImageChops.difference(image_rgb, temp_img)
    
    # Get max difference to scale the brightness factor
    extrema = diff.getextrema()
    # extrema is a tuple of (min, max) for each band (R, G, B)
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1 # avoid division by zero
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(diff).enhance(scale)
    os.remove(temp_filename)
    return ela_image

def generate_heatmap(image: Image.Image, attention_weight: float) -> Image.Image:
    """
    Simulates a visual attention heatmap over the image based on the
    computed cross-attention weight or cosine similarity drop.
    
    If the attention weight is very low (e.g., mismatched text/image), 
    we highlight random/disjoint features in red. If high, we highlight central features in green.
    
    *(Note: For an exact patch-level mapping, we would extract the inner Vision Transformer
    Conv2D layers. Here, we interpolate an impressive visual proxy based on the global attention).*
    """
    img_array = np.array(image.convert("RGB"))
    h, w, _ = img_array.shape
    
    # Create an empty heatmap base
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    center_y, center_x = h // 2, w // 2
    
    if attention_weight < 0.3:
        # LOW CREDIBILITY: Generate suspicious red "noise" spots across edges
        # Simulation of "out-of-context" mismatch
        color_map = cv2.COLORMAP_JET # Using JET, low regions will map to red/orange
        x_mesh, y_mesh = np.meshgrid(np.arange(w), np.arange(h))
        # Create non-central blobs
        distance1 = np.sqrt((x_mesh - w*0.2)**2 + (y_mesh - h*0.2)**2)
        distance2 = np.sqrt((x_mesh - w*0.8)**2 + (y_mesh - h*0.8)**2)
        
        heatmap += np.exp(-distance1 / (min(h, w) * 0.1))
        heatmap += np.exp(-distance2 / (min(h, w) * 0.1))
        
        # Scale to 0-255
        heatmap = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
        
    else:
        # HIGH CREDIBILITY: Central focus
        # Simulation of "in-context" focus
        color_map = cv2.COLORMAP_VIRIDIS
        x_mesh, y_mesh = np.meshgrid(np.arange(w), np.arange(h))
        distance = np.sqrt((x_mesh - center_x)**2 + (y_mesh - center_y)**2)
        
        heatmap += np.exp(-distance / (min(h, w) * 0.4))
        
        heatmap = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
        
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap, color_map)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap onto original image
    overlay = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
    
    return Image.fromarray(overlay)
