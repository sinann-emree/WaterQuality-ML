import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükle
data = pd.read_excel("azaltilmis_dataset.xlsx")

# Histogram
plt.figure(figsize=(12, 8))
data.hist(bins=20, figsize=(15, 10), color='blue', edgecolor='black')
plt.suptitle("Histogramlar", fontsize=16)
plt.tight_layout()
plt.show()

# Boxplot
plt.figure(figsize=(12, 8))
sns.boxplot(data=data, palette="Set2")
plt.title("Boxplot (Kutu Grafikleri)", fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.tight_layout()
plt.show()

