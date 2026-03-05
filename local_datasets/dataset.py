import torch
from torch.utils.data import Dataset
from PIL import Image

class MedicalVLMDataset(Dataset):
    def __init__(self, dataframe, vit_processor, bert_tokenizer, llama_tokenizer, max_length=128):
        self.data = dataframe
        self.vit_processor = vit_processor
        self.bert_tokenizer = bert_tokenizer
        self.llama_tokenizer = llama_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 1. GÖRÜNTÜ İŞLEME (ViT için)
        image = Image.open(row['image_path']).convert("RGB")
        vit_inputs = self.vit_processor(images=image, return_tensors="pt")
        pixel_values = vit_inputs.pixel_values.squeeze(0) # [3, 224, 224]

        # 2. SORU İŞLEME (BERT için)
        question = str(row['question'])
        bert_inputs = self.bert_tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # 3. CEVAP İŞLEME (Llama için Target)
        # Llama'nın ne üretmesi gerektiğini ona öğretiyoruz
        answer = str(row['answer'])
        llama_inputs = self.llama_tokenizer(
            answer + self.llama_tokenizer.eos_token,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "pixel_values": pixel_values,
            "bert_input_ids": bert_inputs.input_ids.squeeze(0),
            "bert_attention_mask": bert_inputs.attention_mask.squeeze(0),
            "llama_labels": llama_inputs.input_ids.squeeze(0)
        }