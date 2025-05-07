import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# Veri setini yükle
file_path = "balanced_dataset.xlsx"  # Veri seti dosya yolu
data = pd.read_excel(file_path)

# Veri setini bağımlı ve bağımsız değişkenlere ayır
X = data.drop(columns=["Potability"])
y = data["Potability"]

# Eğitim ve test setlerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=221, stratify=y)

# Veriyi normalize et (özellikle SVM, KNN ve Logistic Regression için önemlidir)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Modeli oluştur
log_reg = LogisticRegression(max_iter=1000, random_state=444)
log_reg.fit(X_train_scaled, y_train)

# Logistic Regression için test seti tahminlerini al
y_pred_log_reg = log_reg.predict(X_test_scaled)

# Logistic Regression için karmaşıklık matrisini hesapla ve görselleştir
cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
disp_log_reg = ConfusionMatrixDisplay(confusion_matrix=cm_log_reg, display_labels=log_reg.classes_)
disp_log_reg.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# Logistic Regression için diğer performans metriklerini yazdır
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log_reg))

# KNN Modeli oluştur
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_scaled, y_train)

# KNN için test seti tahminlerini al
y_pred_knn = knn_model.predict(X_test_scaled)

# KNN için karmaşıklık matrisini hesapla ve görselleştir
cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=knn_model.classes_)
disp_knn.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - KNN")
plt.show()

# KNN için diğer performans metriklerini yazdır
print("KNN Classification Report:")
print(classification_report(y_test, y_pred_knn))

# SVM Modeli oluştur
svm_model = SVC(probability=True, random_state=None)
svm_model.fit(X_train_scaled, y_train)

# SVM için test seti tahminlerini al
y_pred_svm = svm_model.predict(X_test_scaled)

# SVM için karmaşıklık matrisini hesapla ve görselleştir
cm_svm = confusion_matrix(y_test, y_pred_svm)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=svm_model.classes_)
disp_svm.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - SVM")
plt.show()

# SVM için diğer performans metriklerini yazdır
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))
