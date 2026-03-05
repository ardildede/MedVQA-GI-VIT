import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTImageProcessor, DistilBertTokenizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Kendi dosyalarımız
from data.data_loading import get_kvasir_data, get_train_val_split
from local_datasets.dataset import KvasirHFDataset
from models.model import ViT_BERT_CoAttention_VQA

# --- 1. GRAFİK ÇİZME VE KAYDETME FONKSİYONU ---
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 18)) 
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues') 
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Olan')
    plt.title('ViT + BERT Confusion Matrix (Isı Haritası)')
    plt.savefig("confusion_matrix.png", dpi=300)
    print("📊 Karmaşıklık matrisi 'confusion_matrix.png' olarak kaydedildi.")

# --- 2. MUHAKEME VE TUTARLILIK ANALİZ FONKSİYONU ---
def analyze_model_logic(all_preds, all_labels, answer_map):
    # ID'leri isimlere geri çevir
    inv_map = {v: k for k, v in answer_map.items()}
    
    results = []
    for p, l in zip(all_preds, all_labels):
        results.append({
            'predict': inv_map[p],
            'actual': inv_map[l]
        })
    
    df_res = pd.DataFrame(results)
    
    # --- FACTUAL CONSISTENCY (OLGUSAL TUTARLILIK) ---
    # Kural: Eğer cevap bir 'boyut' ise, gerçekte bir nesne (polip vb.) olmalı.
    size_labels = ['<5mm', '5-10mm', '11-20mm', '>20mm']
    consistency_errors = df_res[
        (df_res['predict'].isin(size_labels)) & (df_res['actual'] == 'none')
    ]
    
    # --- REASONING QUALITY (MUHAKEME KALİTESİ) ---
    # Kural: Gastroskopi bulguları kolonoskopi bölgeleriyle (cecum vb.) karışmamalı.
    reasoning_errors = df_res[
        (df_res['predict'] == 'cecum') & 
        (df_res['actual'].isin(['gastroscopy', 'oesophagitis', 'z-line']))
    ]
    
    total = len(df_res)
    cons_score = 100 - (len(consistency_errors) / total * 100) if total > 0 else 0
    reason_score = 100 - (len(reasoning_errors) / total * 100) if total > 0 else 0
    
    print("\n" + "="*35)
    print("🧠 MUHAKEME VE TUTARLILIK ANALİZİ")
    print("="*35)
    print(f"✅ Factual Consistency Skoru: %{cons_score:.2f}")
    print(f"🧩 Reasoning Quality Skoru:   %{reason_score:.2f}")
    print(f"Toplam Analiz Edilen Örnek:  {total}")
    print(f"Tespit Edilen Mantık Hatası: {len(consistency_errors) + len(reasoning_errors)}")
    print("="*35 + "\n")

# --- 3. ANA ÇALIŞTIRICI ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- ViT + BERT Modeli Başlatılıyor (Cihaz: {device}) ---")

    # HAZIRLIK
    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    raw_data = get_kvasir_data()
    train_data, val_data = get_train_val_split(raw_data)
    
    answers_list = train_data['answer']
    all_answers = sorted(list(set(str(ans).lower() for ans in answers_list)))
    answer_map = {ans: i for i, ans in enumerate(all_answers)}
    label_names = list(answer_map.keys())

    # DATASET & LOADER
    train_ds = KvasirHFDataset(train_data, answer_map)
    val_ds = KvasirHFDataset(val_data, answer_map)

    def collate_fn(batch):
        img_inputs = feature_extractor(images=[item['image'] for item in batch], return_tensors="pt")
        txt_inputs = tokenizer([item['question'] for item in batch], padding=True, truncation=True, max_length=20, return_tensors="pt")
        labels = torch.stack([item['answer'] for item in batch])
        return img_inputs['pixel_values'], txt_inputs['input_ids'], txt_inputs['attention_mask'], labels

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # MODEL & EĞİTİM
    model = ViT_BERT_CoAttention_VQA(num_classes=len(answer_map)).to(device)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    MODEL_PATH = "vit_bert_model.pth"
    
    if os.path.exists(MODEL_PATH):
        print(f"✅ Model yüklendi: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH), strict=False)
    else:
        print("--- Eğitim Başlıyor ---")
        for epoch in range(3):
            model.train()
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for pv, ii, am, lb in loop:
                pv, ii, am, lb = pv.to(device), ii.to(device), am.to(device), lb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(pv, ii, am), lb)
                loss.backward()
                optimizer.step()
        torch.save(model.state_dict(), MODEL_PATH)

    # TEST AŞAMASI
    print("\n--- Test Aşaması ---")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for pv, ii, am, lb in tqdm(val_loader):
            outputs = model(pv.to(device), ii.to(device), am.to(device))
            all_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
            all_labels.extend(lb.numpy())

    # RAPORLAMA (SUPPORT=0 FİLTRELEME)
    present_label_ids = sorted(list(set(all_labels)))
    present_label_names = [label_names[i] for i in present_label_ids]

    print("\n--- Raporlar Oluşturuluyor ---")
    report_text = classification_report(all_labels, all_preds, labels=present_label_ids, target_names=present_label_names, zero_division=0)
    with open("sonuc_raporu.txt", "w", encoding="utf-8") as f: f.write(report_text)
    
    report_dict = classification_report(all_labels, all_preds, labels=present_label_ids, target_names=present_label_names, zero_division=0, output_dict=True)
    pd.DataFrame(report_dict).transpose().to_csv("detayli_analiz.csv")
    
    # MUHAKEME ANALİZİNİ ÇALIŞTIR
    analyze_model_logic(all_preds, all_labels, answer_map)
    
    # MATRİS ÇİZ
    plot_confusion_matrix(all_labels, all_preds, label_names)
    print("🎉 TÜM İŞLEMLER TAMAMLANDI!")

if __name__ == "__main__":
    main()