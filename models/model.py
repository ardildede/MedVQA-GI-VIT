import torch
import torch.nn as nn
from transformers import ViTModel, DistilBertModel

class CoAttentionFusion(nn.Module):
    """
    Görüntü ve metin özelliklerini birbirine bağlayan Cross-Attention katmanı.
    Görüntüyü 'Query', Metni ise 'Key' ve 'Value' olarak kullanır.
    """
    def __init__(self, embed_dim, num_heads=8):
        super(CoAttentionFusion, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, img_seq, text_seq):
        # Cross-attention: Görüntü (Query) metne (Key/Value) bakar
        attn_output, _ = self.multihead_attn(query=img_seq, key=text_seq, value=text_seq)
        # Residual connection ve Normalizasyon
        combined = self.norm(img_seq + attn_output)
        # Sekansın ortalamasını alarak tek bir vektöre indirge
        return combined.mean(dim=1) 

class ViT_BERT_CoAttention_VQA(nn.Module):
    def __init__(self, num_classes):
        super(ViT_BERT_CoAttention_VQA, self).__init__()
        
        # 1. GÖRÜNTÜ: ViT (Patch çıktılarını almak için)
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        
        # 2. METİN: DistilBERT (Token çıktılarını almak için)
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # 3. FUSION: Co-Attention Katmanı
        self.fusion = CoAttentionFusion(embed_dim=768, num_heads=8)
        
        # 4. SINIFLANDIRICI (MLP)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        # ViT Çıktısı: [Batch, 197, 768] (Görüntü yamaları)
        vit_out = self.vit(pixel_values=pixel_values)
        img_seq = vit_out.last_hidden_state 
        
        # BERT Çıktısı: [Batch, Seq_Len, 768] (Kelimeler)
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_seq = bert_out.last_hidden_state
        
        # --- Co-Attention Fusion ---
        fused_features = self.fusion(img_seq, text_seq)
        
        # --- Sınıflandırma ---
        output = self.classifier(fused_features)
        return output