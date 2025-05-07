# Gerekli kütüphaneleri yükle
import pandas as pd
from sklearn.model_selection import train_test_split

# Veri setini yükle
file_path = 'processed_combined_dataset_rounded.xlsx'  # Dosya yolunu uygun şekilde düzenleyin
df = pd.read_excel(file_path)

# %98 oranında veri azaltma
reduction_ratio = 0.98
target_sample_size = int(len(df) * (1 - reduction_ratio))

# Stratified sampling ile sınıf dağılımını koruyarak veri azaltma
df_reduced, _ = train_test_split(
    df, 
    train_size=target_sample_size, 
    stratify=df['Potability'], 
    random_state=42
)

# Azaltılmış veri setinin boyutunu ve sınıf dağılımını kontrol et
reduced_info = df_reduced['Potability'].value_counts(normalize=True)
original_info = df['Potability'].value_counts(normalize=True)

# Sonuçları yazdır
print(f"Orijinal veri seti boyutu: {len(df)}")
print(f"Azaltılmış veri seti boyutu: {len(df_reduced)}")
print("\nOrijinal sınıf dağılımı:")
print(original_info)
print("\nAzaltılmış sınıf dağılımı:")
print(reduced_info)

# Yeni veri setini kaydet
output_path = 'azaltilmis_dataset.xlsx'
df_reduced.to_excel(output_path, index=False)
print(f"\nYeni veri seti başarıyla kaydedildi: {output_path}")

