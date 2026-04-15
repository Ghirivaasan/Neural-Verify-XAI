import torch
import torch.nn as nn
from transformers import CLIPModel

class CrossAttentionConsistency(nn.Module):
    """
    Computes Cross-Attention between Text and Image features to discover
    fine-grained alignment between the modalities.
    """
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, text_features, image_features):
        # text_features: (batch, embed_dim)
        # image_features: (batch, embed_dim)
        
        # We need sequence dimension for MultiheadAttention
        # Shape: (batch, 1, embed_dim)
        q = text_features.unsqueeze(1)
        k = image_features.unsqueeze(1)
        v = image_features.unsqueeze(1)
        
        # attn_output: (batch, 1, embed_dim)
        # attn_weights: (batch, 1, 1)
        attn_output, attn_weights = self.multihead_attn(q, k, v)
        
        # Add & Norm
        output = self.layer_norm(q + attn_output)
        
        # Squeeze back to (batch, embed_dim)
        return output.squeeze(1), attn_weights.squeeze(1)


class MultimodalFakeNewsModel(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", hidden_dim=512, dropout=0.3):
        super().__init__()
        # Load pre-trained CLIP model
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        
        # Freeze CLIP parameters to prevent catastrophic forgetting and speed up training,
        # or keep them trainable for fine-tuning. Here we freeze the base model and only
        # train the fusion and classification heads.
        for param in self.clip.parameters():
            param.requires_grad = False
            
        embed_dim = self.clip.config.projection_dim
        
        # Consistency Module
        self.cross_attention = CrossAttentionConsistency(embed_dim=embed_dim)
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        # Classification Head (MLP)
        # Input features:
        # 1. Text embedding (embed_dim)
        # 2. Image embedding (embed_dim)
        # 3. Cross-attention output (embed_dim)
        # 4. Cosine similarity score (1)
        # Total: embed_dim * 3 + 1
        concat_dim = (embed_dim * 3) + 1
        
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1) # Raw logits for BCEWithLogitsLoss
        )

    def forward(self, input_ids, attention_mask, pixel_values, return_attention=False):
        # Extract Text and Image features using CLIP
        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        
        # Get L2-normalized embeddings
        text_embeds = outputs.text_embeds 
        image_embeds = outputs.image_embeds 
        
        # 1. Compute Cross-Attention Consistency
        cross_attn_out, attn_weights = self.cross_attention(text_embeds, image_embeds)
        
        # 2. Compute basic Cosine Similarity
        cos_sim = self.cosine_similarity(text_embeds, image_embeds).unsqueeze(1) # Shape: (batch, 1)
        
        # 3. Concatenate all features
        # Shape: (batch, embed_dim * 3 + 1)
        fused_features = torch.cat([
            text_embeds,
            image_embeds,
            cross_attn_out,
            cos_sim
        ], dim=1)
        
        # 4. Classification
        logits = self.classifier(fused_features)
        
        if return_attention:
            return logits.squeeze(-1), cos_sim, attn_weights
        return logits.squeeze(-1) # Shape: (batch,)

    def predict_score(self, input_ids, attention_mask, pixel_values, return_attention=False):
        """
        """
        if return_attention:
            logits, cos_sim, attn_weights = self.forward(input_ids, attention_mask, pixel_values, return_attention=True)
            scores = torch.sigmoid(logits)
            return scores, cos_sim, attn_weights
            
        logits = self.forward(input_ids, attention_mask, pixel_values)
        return torch.sigmoid(logits)
