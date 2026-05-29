%%writefile main.py
import os
import torch
import pandas as pd
from unsloth import FastLanguageModel
from transformers import ViTImageProcessor, BertTokenizer, Trainer, TrainingArguments

# Modüler Dosyalarımızdan Sınıfları Çağırıyoruz
from data.data_loading import get_kvasir_data_split
from local_datasets.dataset import MedicalVLMDataset
from models.model import ViT_BERT_Llama_CrossAttention_VLM
from utils.metrics import get_compute_metrics_fn

def main():
    max_seq_length = 512
    load_in_4bit = True 
    
    print("🚀 Modeller hafızaya yükleniyor (Llama 3.1 + ViT + BERT)...")
    
    # 1. LLAMA MODEL KURULUMU
    llama_model, llama_tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
    )

    llama_model = FastLanguageModel.get_peft_model(
        llama_model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
    )

    # 2. TOKEnIZER VE PROCESSOR AYARLARI
    vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    llama_tokenizer.pad_token = llama_tokenizer.eos_token

    # 3. HİBRİT CROSS-ATTENTION MODELİNİN OLUŞTURULMASI
    model = ViT_BERT_Llama_CrossAttention_VLM(llama_model, llama_tokenizer).to("cuda")

    # 4. VERİ SETİNİN BÖLÜNMESİ VE YÜKLENMESİ
    train_df, val_df = get_kvasir_data_split("data/kvasir_vqa.csv")
    
    train_ds = MedicalVLMDataset(train_df, vit_processor, bert_tokenizer, llama_tokenizer)
    val_ds = MedicalVLMDataset(val_df, vit_processor, bert_tokenizer, llama_tokenizer)

    # 5. EĞİTİM PARMETRELERİ
    training_args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 50, 
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 5,
        output_dir = "outputs",
        
        # Doğrulama ayarları
        evaluation_strategy = "steps",
        eval_steps = 10,
        per_device_eval_batch_size = 2,
        logging_dir = "./logs",
        save_total_limit = 1,
    )

    # 6. TRAINER VE METRİK BAĞLANTISI
    compute_metrics_fn = get_compute_metrics_fn(llama_tokenizer)
    
    print("⚡ Eğitim döngüsü başlatılıyor...")
    trainer = Trainer(
        model = model,
        train_dataset = train_ds,
        eval_dataset = val_ds,
        args = training_args,
        compute_metrics = compute_metrics_fn
    )
    
    trainer.train()

    # 7. MODEL KAYDETME SÜRECİ
    print("💾 Sadece eğitilen adaptörler ve köprüler kaydediliyor...")
    model_save_path = "medical_vlm_cross_attn"
    os.makedirs(model_save_path, exist_ok=True)
    
    torch.save(model.vision_projector.state_dict(), f"{model_save_path}/vision_proj.bin")
    torch.save(model.bert_projector.state_dict(), f"{model_save_path}/bert_proj.bin")
    torch.save(model.cross_attention.state_dict(), f"{model_save_path}/cross_attention.bin")
    llama_model.save_pretrained(f"{model_save_path}/llama_lora")
    
    print(f"✅ Başarılı! Tüm ağırlıklar '{model_save_path}' klasörüne atıldı.")

if __name__ == "__main__":
    main()