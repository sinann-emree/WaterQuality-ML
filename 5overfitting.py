import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

# Veri setini yükle
data = pd.read_excel("azaltilmis_dataset.xlsx")

# Veri setini bağımlı ve bağımsız değişkenlere ayır
X = data.drop(columns=["Potability"])
y = data["Potability"]

# Eğitim ve test setini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi normalize et (özellikle SVM, KNN ve Logistic Regression için önemlidir)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# K-Fold Cross Validation (5 katlı)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ------------------- K-Nearest Neighbors (KNN) Modeli -------------------

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_cv_scores = cross_val_score(knn_model, X_train_scaled, y_train, cv=kf, scoring='accuracy')
print("KNN Cross Validation Scores:", knn_cv_scores)
print("KNN Average Cross Validation Accuracy:", np.mean(knn_cv_scores))

knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
y_pred_knn_proba = knn_model.predict_proba(X_test_scaled)[:, 1]

print("\nKNN Results:")
print("Accuracy (Train):", np.mean(knn_cv_scores))  
print("Accuracy (Test):", accuracy_score(y_test, y_pred_knn))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_knn_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))

train_accuracy_knn = np.mean(knn_cv_scores)
test_accuracy_knn = accuracy_score(y_test, y_pred_knn)
if train_accuracy_knn > test_accuracy_knn :
    print("KNN Model: Overfitting tespit edildi")
else:
    print("KNN Model: Overfitting tespit edilemedi")

# ROC eğrisini çiz
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_pred_knn_proba)
plt.plot(fpr_knn, tpr_knn, color='b', label='KNN ROC Curve')

# ------------------- Support Vector Machine Modeli -------------------
# SVM Modeli oluştur
svm_model = SVC(probability=True, random_state=42)

# K-Fold Cross Validation ile eğitim ve doğrulama
svm_cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=kf, scoring='accuracy')
print("\nSVM Cross Validation Scores:", svm_cv_scores)
print("SVM Average Cross Validation Accuracy:", np.mean(svm_cv_scores))

# Eğitim ve test doğruluğunu ölç
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
y_pred_svm_proba = svm_model.predict_proba(X_test_scaled)[:, 1]

print("\nSVM Results:")
print("Accuracy (Train):", np.mean(svm_cv_scores))  # Eğitim doğruluğu (K-fold ortalaması)
print("Accuracy (Test):", accuracy_score(y_test, y_pred_svm))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_svm_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))

# Overfitting analizi: Eğitim ve test doğruluk oranları arasındaki fark
train_accuracy_svm = np.mean(svm_cv_scores)
test_accuracy_svm = accuracy_score(y_test, y_pred_svm)
if train_accuracy_svm > test_accuracy_svm :
    print("SVM Model: Overfitting tespit edildi")
else:
    print("SVM Model: Overfitting tespit edilemedi")

# ROC eğrisini çiz
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_svm_proba)
plt.plot(fpr_svm, tpr_svm, color='r', label='SVM ROC Curve')

# ------------------- Logistic Regression Modeli -------------------
# Logistic Regression Modeli oluştur
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# K-Fold Cross Validation ile eğitim ve doğrulama
log_reg_cv_scores = cross_val_score(log_reg, X_train_scaled, y_train, cv=kf, scoring='accuracy')
print("\nLogistic Regression Cross Validation Scores:", log_reg_cv_scores)
print("Logistic Regression Average Cross Validation Accuracy:", np.mean(log_reg_cv_scores))

# Eğitim ve test doğruluğunu ölç
log_reg.fit(X_train_scaled, y_train)
y_pred_logreg = log_reg.predict(X_test_scaled)
y_pred_logreg_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

print("\nLogistic Regression Results:")
print("Accuracy (Train):", np.mean(log_reg_cv_scores))  # Eğitim doğruluğu (K-fold ortalaması)
print("Accuracy (Test):", accuracy_score(y_test, y_pred_logreg))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_logreg_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred_logreg))

# Overfitting analizi: Eğitim ve test doğruluk oranları arasındaki fark
train_accuracy_logreg = np.mean(log_reg_cv_scores)
test_accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
if train_accuracy_logreg > test_accuracy_logreg :
    print("Logistic Regression Model: Overfitting tespit edildi")
else:
    print("Logistic Regression Model: Overfitting tespit edilemedi")

# ROC eğrisini çiz
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_pred_logreg_proba)
plt.plot(fpr_logreg, tpr_logreg, color='g', label='Logistic Regression ROC Curve')

# Son olarak ROC eğrisini göster
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Rastgele tahmin çizgisi
plt.title('ROC Eğrisi')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()
