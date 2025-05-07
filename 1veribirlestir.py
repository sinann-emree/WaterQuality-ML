import pandas as pd

# Excel dosyalarını okuma
df1 = pd.read_excel('water_potability.xlsx')  # İlk veri seti
df2 = pd.read_excel('Watera.xlsx')  # İkinci veri seti

# Birleştirme işlemi
# Satır bazlı birleştirme (üst üste eklemek için)
combined_df = pd.concat([df1, df2], ignore_index=True)

# Sütun bazlı birleştirme (ortak bir sütun üzerinden birleştirme)
# combined_df = pd.merge(df1, df2, on='ortak_sütun_adi', how='inner')  # 'inner', 'outer', 'left', 'right'

# Birleştirilen veri setini kaydetme
combined_df.to_excel('combined_dataset.xlsx', index=False)

print("Veri setleri birleştirildi ve 'combined_dataset.xlsx' olarak kaydedildi.")
