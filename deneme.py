import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

# Veri setini yükle
data = pd.read_excel("balanced_dataset.xlsx")

# Veri setini bağımlı ve bağımsız değişkenlere ayır
X = data.drop(columns=["Potability"])
y = data["Potability"]

# Eğitim ve test setini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi normalize et
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# K-Fold Cross Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Random Forest ile öznitelik önemlerini belirle
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)
feature_importances = rf_model.feature_importances_
important_features = pd.DataFrame({"Feature": X.columns, "Importance": feature_importances}).sort_values(by="Importance", ascending=False)
print("\nFeature Importances:\n", important_features)

# Hiperparametre araması için GridSearchCV parametreleri
param_grids = {
    "knn": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"]
    },
    "svm": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    },
    "logistic_regression": {
        "C": [0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"]
    }
}

# Hiperparametre optimizasyonu ve model oluşturma
optimized_models = {}

# KNN
knn = KNeighborsClassifier()
gs_knn = GridSearchCV(knn, param_grids["knn"], cv=kf, scoring='accuracy')
gs_knn.fit(X_train_scaled, y_train)
optimized_models["knn"] = gs_knn.best_estimator_
print("\nBest Parameters for KNN:", gs_knn.best_params_)

# SVM
svm = SVC(probability=True, random_state=42)
gs_svm = GridSearchCV(svm, param_grids["svm"], cv=kf, scoring='accuracy')
gs_svm.fit(X_train_scaled, y_train)
optimized_models["svm"] = gs_svm.best_estimator_
print("\nBest Parameters for SVM:", gs_svm.best_params_)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
gs_log_reg = GridSearchCV(log_reg, param_grids["logistic_regression"], cv=kf, scoring='accuracy')
gs_log_reg.fit(X_train_scaled, y_train)
optimized_models["logistic_regression"] = gs_log_reg.best_estimator_
print("\nBest Parameters for Logistic Regression:", gs_log_reg.best_params_)

# Modelleri değerlendirme ve karşılaştırma
results = {}

for name, model in optimized_models.items():
    print(f"\n{name.upper()} Results:")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    train_accuracy = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='accuracy').mean()
    test_accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("Accuracy (Train):", train_accuracy)
    print("Accuracy (Test):", test_accuracy)
    print("ROC-AUC Score:", roc_auc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # ROC eğrisini çizme
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name.upper()} (AUC={roc_auc:.2f})')

    results[name] = {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "roc_auc": roc_auc
    }

# Son olarak ROC eğrisini göster
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC Eğrisi')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Sonuçların özeti
results_df = pd.DataFrame(results).T
print("\nModel Performance Summary:\n", results_df)
