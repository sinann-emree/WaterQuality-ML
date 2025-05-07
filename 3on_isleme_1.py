import pandas as pd

# Dosya yolunu belirt
file_path = 'combined_dataset_rounded.xlsx'

# Veri setini yükle
df = pd.read_excel(file_path)

# Eksik değerleri doldurma stratejileri
# 1. pH: Medyan ile doldur
df['ph'] = df['ph'].fillna(df['ph'].median())

# 2. Hardness: Ortalama ile doldur
df['Hardness'] = df['Hardness'].fillna(df['Hardness'].mean())

# 3. Solids: Ortalama ile doldur
df['Solids'] = df['Solids'].fillna(df['Solids'].mean())

# 4. Chloramines: Ortalama ile doldur
df['Chloramines'] = df['Chloramines'].fillna(df['Chloramines'].mean())

# 5. Sulfate: Ortalama ile doldur
df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].mean())

# 6. Conductivity: Ortalama ile doldur
df['Conductivity'] = df['Conductivity'].fillna(df['Conductivity'].mean())

# 7. Organic Carbon: Ortalama ile doldur
df['Organic_carbon'] = df['Organic_carbon'].fillna(df['Organic_carbon'].mean())

# 8. Trihalomethanes: Medyan ile doldur
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df['Trihalomethanes'].median())

# 9. Turbidity: Ortalama ile doldur
df['Turbidity'] = df['Turbidity'].fillna(df['Turbidity'].mean())

# 10. Potability: Eksik değer içeren satırları çıkar
df = df.dropna(subset=['Potability'])

# İşlenmiş veri setini yeni bir dosyaya kaydet
output_file_path = 'processed_combined_dataset_rounded.xlsx'
df.to_excel(output_file_path, index=False)

# İşlem tamamlandığında kullanıcıya bilgi ver
print(f"İşlenmiş veri seti '{output_file_path}' olarak kaydedildi.")
