import pandas as pd
import requests
import os
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import time

def download_fakeddit_subset(tsv_path, output_dir="datasets/fakeddit_subset", max_samples=500, delay=0.1):
    """
    Parses a Fakeddit TSV file, downloads the associated images,
    and creates a clean CSV for our PyTorch Dataset.
    
    Expected TSV Columns: ['author', 'clean_title', 'created_utc', 'domain', 'hasImage', 'id', 'image_url', 'linked_submission_id', 'num_comments', 'score', 'subreddit', 'title', 'upvote_ratio', '2_way_label', '3_way_label', '6_way_label']
    """
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"Loading TSV from {tsv_path}...")
    try:
        df = pd.read_csv(tsv_path, sep='\t')
    except Exception as e:
        print(f"Error loading TSV: {e}")
        return None
        
    print(f"Total rows found: {len(df)}")
    
    # Filter for rows that actually have images
    if 'hasImage' in df.columns:
        df = df[df['hasImage'] == True]
    elif 'image_url' in df.columns:
        df = df[df['image_url'].notna()]
        
    # We only care about: image_url, clean_title (text), and 2_way_label (0=Fake, 1=True)
    # Note: Fakeddit 2_way_label usually has 1 as fake and 0 as true, 
    # but let's standardize to our format: 1=Credible(True), 0=Fake.
    # In Fakeddit: 0 is true, 1 is fake. We will invert it.
    
    valid_data = []
    success_count = 0
    
    print(f"Downloading up to {max_samples} images...")
    for idx, row in tqdm(df.iterrows(), total=min(len(df), max_samples*3)):
        if success_count >= max_samples:
            break
            
        img_url = str(row.get('image_url', ''))
        text = str(row.get('clean_title', row.get('title', '')))
        
        # Fakeddit: 0 = True, 1 = Fake. We want 1 = True, 0 = Fake.
        fakeddit_label = row.get('2_way_label', 0)
        our_label = 1 if fakeddit_label == 0 else 0
        
        if not img_url.startswith("http"):
            continue
            
        img_filename = f"{row.get('id', success_count)}.jpg"
        img_path = os.path.join(images_dir, img_filename)
        
        # Download image if it doesn't exist
        if not os.path.exists(img_path):
            try:
                response = requests.get(img_url, timeout=5)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                    img.thumbnail((512, 512)) # Save space
                    img.save(img_path)
                    time.sleep(delay) # Be polite
                else:
                    continue # Skip if URL is dead
            except Exception:
                continue # Skip on timeout/error
                
        valid_data.append({
            'image_path': os.path.abspath(img_path),
            'text': text,
            'label': our_label
        })
        success_count += 1
        
    final_df = pd.DataFrame(valid_data)
    csv_out = os.path.join(output_dir, "clean_dataset.csv")
    final_df.to_csv(csv_out, index=False)
    
    print(f"Successfully processed {success_count} samples!")
    print(f"Dataset saved to: {csv_out}")
    return final_df

if __name__ == "__main__":
    print("Fakeddit Parser Module Helper")
    print("To use: call download_fakeddit_subset('path/to/multimodal_train.tsv')")
