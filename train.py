import torch
from unsloth import FastLanguageModel
from transformers import ViTImageProcessor, BertTokenizer, Trainer, TrainingArguments
from models.medical_vlm import ViT_BERT_Llama_VLM
from local_datasets.vlm_dataset import MedicalVLMDataset
import pandas as pd

# 1. KONFİGÜRASYON VE MODELLERİN YÜKLENMESİ
max_seq_length = 128 
dtype = None # Auto detection
load_in_4bit = True # Bellek tasarrufu için kritik

# Unsloth ile Llama'yı yükle
llama_model, llama_tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit,
)

# LoRA Adaptörlerini ekle (Sadece Llama'nın %1'ini eğiteceğiz)
llama_model = FastLanguageModel.get_peft_model(
    llama_model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)

# 2. HİBRİT MODELİ OLUŞTUR
# Kendi yazdığımız sınıfı çağırıyoruz
model = ViT_BERT_Llama_VLM(llama_model, llama_tokenizer).to("cuda")

# İşlemcileri ve Tokenizerları hazırla
vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
llama_tokenizer.pad_token = llama_tokenizer.eos_token # Padding ayarı

# 3. VERİ SETİNİ HAZIRLA
# Veri setinin csv/dataframe olduğunu varsayıyoruz
df = pd.read_csv("kvasir_vqa.csv") 
train_ds = MedicalVLMDataset(df, vit_processor, bert_tokenizer, llama_tokenizer)

# 4. EĞİTİM ARGÜMANLARI (Akademik Standartlarda)
training_args = TrainingArguments(
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    max_steps = 100, # Test için kısa tutuldu, gerçek eğitimde artırılmalı
    learning_rate = 2e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 1,
    output_dir = "outputs",
    save_strategy = "steps",
    save_steps = 50,
)

# 5. EĞİTİMİ BAŞLAT
print("--- Hibrit VLM Eğitimi Başlıyor ---")
trainer = Trainer(
    model = model,
    train_dataset = train_ds,
    args = training_args,
)

trainer.train()

# 6. MODELİ KAYDET
# Sadece eğittiğimiz Projection katmanlarını ve LoRA ağırlıklarını kaydeder
torch.save(model.vision_projector.state_dict(), "vision_projector.bin")
torch.save(model.bert_projector.state_dict(), "bert_projector.bin")
llama_model.save_pretrained("llama_lora_medical")
print("🎉 Model başarıyla eğitildi ve kaydedildi!")