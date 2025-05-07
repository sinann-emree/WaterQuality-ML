# Gerekli kütüphaneleri içe aktar
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükle
data = pd.read_excel("azaltilmis_dataset.xlsx")

# Korelasyon analizi
correlation_matrix = data.corr()
correlation_with_target = correlation_matrix["Potability"].abs()  # Korelasyonun mutlak değerini al

# Düşük korelasyonlu sütunları belirle (eşik değeri 0.05 olarak alınmıştır)
low_correlation_features = correlation_with_target[correlation_with_target < 0.05].index

# Düşük korelasyonlu sütunları çıkar
filtered_data = data.drop(columns=low_correlation_features)

# Sonuçları göster
print("Düşük korelasyonlu sütunlar:", list(low_correlation_features))
print("\nAzaltılmış veri setinin sütunları:", list(filtered_data.columns))

# Azaltılmış veri setini kaydetmek isterseniz
filtered_data.to_excel("filtered_dataset.xlsx", index=False)

# 1. Korelasyon matrisini görselleştir
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Korelasyon Matrisi")
plt.tight_layout()
plt.show()

# 2. Potability hedef değişkeni ile olan korelasyonu bar grafikte göster
plt.figure(figsize=(10, 6))
sns.barplot(x=correlation_with_target.index, y=correlation_with_target.values, palette="viridis")
plt.title("Potability ile Korelasyon - Bar Plot")
plt.xticks(rotation=90)  # Etiketlerin düzgün görünmesi için döndür
plt.tight_layout()
plt.show()
