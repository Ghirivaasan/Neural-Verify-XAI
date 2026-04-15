import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm

def evaluate_model(model, test_loader, device='cuda', save_dir="results"):
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_scores = []
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("Evaluating model...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].cpu().numpy()
            
            scores = model.predict_score(input_ids, attention_mask, pixel_values).cpu().numpy()
            preds = (scores > 0.5).astype(int)
            
            all_scores.extend(scores)
            all_preds.extend(preds)
            all_labels.extend(labels)
            
    all_scores = np.array(all_scores)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except ValueError:
        auc = 0.5 # Default if only one class is present in mock data
    
    metrics = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc
    }
    
    print("\n--- Evaluation Results ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    # Generate Confusion Matrix Figure
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion Matrix saved to {cm_path}")
    
    return metrics

if __name__ == "__main__":
    # Smoke test
    from dataset import MultimodalFakeNewsDataset, get_mock_dataframe
    from transformers import CLIPProcessor
    from model import MultimodalFakeNewsModel
    from torch.utils.data import DataLoader
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize a mock model and data
    model = MultimodalFakeNewsModel()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    df_test = get_mock_dataframe(20)
    test_dataset = MultimodalFakeNewsDataset(df_test, processor)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    evaluate_model(model, test_loader, device=device)
