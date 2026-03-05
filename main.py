import torch
from unsloth import FastLanguageModel
from transformers import ViTImageProcessor, BertTokenizer, Trainer, TrainingArguments
from models.model import ViT_BERT_Llama_VLM  
from local_datasets.dataset import MedicalVLMDataset 
import pandas as pd
import os

def main():
    # 1. KONFİGÜRASYON
    max_seq_length = 2048 
    load_in_4bit = True 
    
    print("🚀 Modeller yükleniyor (Llama 3.1 + ViT + BERT)...")
    
    # 2. LLAMA YÜKLEME (Unsloth)
    llama_model, llama_tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
    )

    # Llama'ya LoRA adaptörlerini ekle
    llama_model = FastLanguageModel.get_peft_model(
        llama_model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
    )

    # 3. HİBRİT VLM MODELİNİ OLUŞTUR
    # model.py dosyasındaki ViT_BERT_Llama_VLM sınıfını kullanıyoruz
    model = ViT_BERT_Llama_VLM(llama_model, llama_tokenizer).to("cuda")

    # Processor ve Tokenizer'ları hazırla
    vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    llama_tokenizer.pad_token = llama_tokenizer.eos_token

    # 4. VERİ SETİ (CSV dosyanın yolunu buraya yaz)
    # csv dosyasında 'image_path', 'question' ve 'answer' sütunları olmalı
    if os.path.exists("data/kvasir_vqa.csv"):
        df = pd.read_csv("data/kvasir_vqa.csv")
        train_ds = MedicalVLMDataset(df, vit_processor, bert_tokenizer, llama_tokenizer)
    else:
        print("❌ Hata: 'data/kvasir_vqa.csv' bulunamadı!")
        return

    # 5. EĞİTİM AYARLARI
    training_args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 60, # İlk deneme için kısa tutuldu
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
        save_total_limit = 2,
    )

    # 6. EĞİTİMİ BAŞLAT
    print("⚡ Eğitim başlıyor...")
    trainer = Trainer(
        model = model,
        train_dataset = train_ds,
        args = training_args,
    )
    
    trainer.train()

    # 7. KAYDETME
    print("💾 Model kaydediliyor...")
    model_save_path = "medical_vlm_final"
    # Sadece eğitilen kısımları (Projection ve LoRA) kaydetmek yeterli
    torch.save(model.vision_projector.state_dict(), f"{model_save_path}/vision_proj.bin")
    torch.save(model.bert_projector.state_dict(), f"{model_save_path}/bert_proj.bin")
    llama_model.save_pretrained(f"{model_save_path}/llama_lora")
    
    print(f"✅ İşlem tamamlandı! Model '{model_save_path}' klasörüne kaydedildi.")

if __name__ == "__main__":
    main()