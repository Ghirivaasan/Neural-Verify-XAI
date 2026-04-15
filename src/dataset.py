import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from transformers import CLIPProcessor

class MultimodalFakeNewsDataset(Dataset):
    """
    Dataset for loading Images and Text for Multimodal Fake News Detection.
    Expects a DataFrame with columns: ['image_path', 'text', 'label']
    - label: 1 for Real (Credible), 0 for Fake (Low Credibility)
    """
    
    def __init__(self, metadata_df, processor: CLIPProcessor, base_image_dir="", max_length=77):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing 'image_path', 'text', 'label'.
            processor (CLIPProcessor): Pre-trained CLIP processor.
            base_image_dir (str): Base directory for image paths, if absolute paths are not provided.
            max_length (int): Maximum token length for the text processor.
        """
        self.data = metadata_df.reset_index(drop=True)
        self.processor = processor
        self.base_image_dir = base_image_dir
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['text'])
        label = torch.tensor(row['label'], dtype=torch.float32)

        # Handle image loading
        img_path = str(row['image_path'])
        if self.base_image_dir and not os.path.isabs(img_path):
            img_path = os.path.join(self.base_image_dir, img_path)
            
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback to a zero-tensor image or placeholder if image is missing/corrupted
            # In a real scenario, we might want to drop these rows beforehand
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        # Process text and image using CLIP Processor
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        # The processor returns batch structures, we squeeze them for the Dataset item
        item = {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'label': label
        }

        return item

def get_mock_dataframe(num_samples=100):
    """Helper function to generate a mock dataset for testing purposes."""
    import numpy as np
    data = []
    for i in range(num_samples):
        # Fake data
        data.append({
            'image_path': f"mock_img_{i}.jpg",
            'text': f"This is a placeholder breaking news headline {i}.",
            'label': float(np.random.randint(0, 2))
        })
    return pd.DataFrame(data)
