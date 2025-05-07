import pandas as pd

# combined_dataset.xlsx dosyasını oku
df = pd.read_excel('combined_dataset.xlsx')

# Sayısal sütunları belirle (tüm sayısal sütunları alıyoruz)
numeric_columns = df.select_dtypes(include='number').columns

# Virgülden sonra iki basamağa yuvarlama
df[numeric_columns] = df[numeric_columns].round(2)

# Sonuçları kontrol et
print(df.head())

# Yuvarlanmış veriyi kaydet (isteğe bağlı)
df.to_excel('combined_dataset_rounded.xlsx', index=False)
