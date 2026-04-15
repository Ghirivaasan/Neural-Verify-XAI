import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=5, device='cuda', save_dir="checkpoints"):
    model.to(device)
    os.makedirs(save_dir, exist_ok=True)
    
    # Using mixed precision for faster training and less memory usage
    scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda'))
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    print(f"Starting training on {device} for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                logits = model(input_ids, attention_mask, pixel_values)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['label'].to(device)
                
                with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                    logits = model(input_ids, attention_mask, pixel_values)
                    loss = criterion(logits, labels)
                    
                val_loss += loss.item()
                
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1} Results: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        scheduler.step(avg_val_loss)
        
        # Save exact best model checkpt
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path} with loss {best_val_loss:.4f}")
            
    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f)
        
    return history

if __name__ == "__main__":
    from dataset import MultimodalFakeNewsDataset, get_mock_dataframe
    from transformers import CLIPProcessor
    from model import MultimodalFakeNewsModel
    
    print("Running training test module with mock data...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    df_train = get_mock_dataframe(64)
    df_val = get_mock_dataframe(16)
    
    train_dataset = MultimodalFakeNewsDataset(df_train, processor)
    val_dataset = MultimodalFakeNewsDataset(df_val, processor)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    model = MultimodalFakeNewsModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=2, device=device)
    print("Mock training complete. Excellent!")
