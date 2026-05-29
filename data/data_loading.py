%%writefile data/data_loading.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def get_kvasir_data_split(csv_path="data/kvasir_vqa.csv", test_size=0.2):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        print(f"⚠️ {csv_path} bulunamadı, Kaggle/Colab için geçici sanal veri üretiliyor...")
        os.makedirs("data", exist_ok=True)
        from PIL import Image
        img = Image.new('RGB', (224, 224), color = 'blue')
        img.save('data/dummy.jpg')
        df = pd.DataFrame({
            'image_path': ['data/dummy.jpg'] * 10,
            'question': ['Is there any abnormality?'] * 10,
            'answer': ['Yes, a lesion is noted.'] * 10
        })
        df.to_csv(csv_path, index=False)
        
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df, val_df