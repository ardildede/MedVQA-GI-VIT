import torch
import torch.nn as nn
from transformers import ViTModel, BertModel, AutoTokenizer
from unsloth import FastLanguageModel

class ViT_BERT_Llama_VLM(nn.Module):
    def __init__(self, llama_model, llama_tokenizer):
        super(ViT_BERT_Llama_VLM, self).__init__()
        
        # 1. BEYİN: Unsloth Llama 3.1
        self.llama = llama_model
        self.tokenizer = llama_tokenizer
        
        # 2. GÖZ: ViT (Görüntüleri anlamak için)
        self.vision_tower = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
        # 3. DİL UZMANI: BERT (Soruyu tıbbi olarak rafine etmek için)
        self.bert_encoder = BertModel.from_pretrained("bert-base-uncased")
        
        # 4. KÖPRÜLER (Projection Layers)
        # ViT çıktı (768) -> Llama giriş (4096)
        self.vision_projector = nn.Sequential(
            nn.Linear(768, 4096),
            nn.GELU(),
            nn.Linear(4096, 4096)
        )
        
        # BERT çıktı (768) -> Llama giriş (4096)
        self.bert_projector = nn.Linear(768, 4096)

        # Eğitim stratejisi: Encoderları dondur, sadece köprüleri ve Llama'nın LoRA katmanlarını eğit
        for param in self.vision_tower.parameters(): param.requires_grad = False
        for param in self.bert_encoder.parameters(): param.requires_grad = False

    def forward(self, images, question_input_ids, question_attention_mask, labels=None):
        # A) Görüntü Özellikleri: [Batch, 197, 768] -> [Batch, 197, 4096]
        with torch.no_grad():
            vit_outputs = self.vision_tower(pixel_values=images)
            image_feats = vit_outputs.last_hidden_state
        image_embeds = self.vision_projector(image_feats)
        
        # B) BERT Soru Özellikleri: [Batch, Seq_Len, 768] -> [Batch, Seq_Len, 4096]
        with torch.no_grad():
            bert_outputs = self.bert_encoder(input_ids=question_input_ids, attention_mask=question_attention_mask)
            question_feats = bert_outputs.last_hidden_state
        question_embeds = self.bert_projector(question_feats)
        
        # C) FÜZYON (Early Fusion): Hepsini bir sıraya diziyoruz
        inputs_embeds = torch.cat((image_embeds, question_embeds), dim=1)
        
        # D) LLAMA ÜRETİM
        outputs = self.llama(
            inputs_embeds=inputs_embeds,
            labels=labels, 
            return_dict=True
        )
        
        return outputs