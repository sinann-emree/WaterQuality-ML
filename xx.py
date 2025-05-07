# Gerekli kütüphaneleri içe aktar
import pandas as pd
from sklearn.utils import resample

# Veri setini yükle
data = pd.read_excel("azaltilmis_dataset.xlsx")

# Potability'si 1 olan azınlık sınıfı ve Potability'si 0 olan çoğunluk sınıfını ayır
minority_class = data[data['Potability'] == 1]
majority_class = data[data['Potability'] == 0]

# Azınlık sınıfını çoğunluk sınıfı ile aynı sayıya getirmek için oversampling yap
minority_class_upsampled = resample(minority_class, 
                                    replace=True,  # Örnekleri tekrar kullanarak artır
                                    n_samples=len(majority_class),  # Çoğunluk sınıfıyla eşit olacak şekilde örnek sayısını artır
                                    random_state=42)  # Tekrarlanabilirlik için sabit bir random state kullan

# Çoğunluk ve artırılmış azınlık sınıflarını birleştir
balanced_data = pd.concat([majority_class, minority_class_upsampled])

# Veri setini karıştır
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Yeni dengeli veri setini kaydet
balanced_data.to_excel("balanced_dataset.xlsx", index=False)

# Yeni veri setinin sınıf dağılımını kontrol et
print(balanced_data['Potability'].value_counts())
