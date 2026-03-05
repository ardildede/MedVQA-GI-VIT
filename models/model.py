%%writefile vqa_project/models/model.py
import torch
import torch.nn as nn
from transformers import ViTModel, DistilBertModel

class ViT_BERT_VQA(nn.Module):
    def __init__(self, num_classes):
        super(ViT_BERT_VQA, self).__init__()
        
        # 1. GÖRÜNTÜ: Vision Transformer (ViT)
        # Google'ın eğittiği base model (ImageNet-21k)
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        
        # 2. METİN: DistilBERT
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Boyutlar: ViT Base = 768, DistilBERT = 768
        combined_dim = 768 + 768 
        
        # 3. SINIFLANDIRICI
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        # --- Görüntü (ViT) ---
        # ViT çıktısı: last_hidden_state (Batch, Seq, 768) ve pooler_output (Batch, 768)
        # pooler_output her zaman kullanılabilir değil, [CLS] tokenini (ilk token) alıyoruz.
        vit_out = self.vit(pixel_values=pixel_values)
        img_feat = vit_out.last_hidden_state[:, 0, :] # [CLS] tokeni (Resmin özeti)
        
        # --- Metin (BERT) ---
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = bert_out.last_hidden_state[:, 0, :] # [CLS] tokeni (Cümlenin özeti)
        
        # --- Birleştir ---
        combined = torch.cat((img_feat, text_feat), dim=1)
        output = self.classifier(combined)
        
        return output