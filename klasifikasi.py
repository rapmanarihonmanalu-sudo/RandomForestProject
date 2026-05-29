import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# ==================================================
# MEMBACA DATASET
# ==================================================

df = pd.read_csv('heart.csv')

# Menampilkan 5 data pertama
print("===== DATASET =====")
print(df.head())

# ==================================================
# EDA (EXPLORATORY DATA ANALYSIS)
# ==================================================

plt.figure(figsize=(6,4))

df['target'].value_counts().plot(kind='bar')

plt.title('Distribusi Target')
plt.xlabel('Kelas')
plt.ylabel('Jumlah Data')

plt.show()

# ==================================================
# MENENTUKAN FEATURE DAN TARGET
# ==================================================

X = df.drop('target', axis=1)
y = df['target']

# ==================================================
# SPLIT DATA TRAINING DAN TESTING
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==================================================
# MODEL 1 : RANDOM FOREST
# ==================================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Training model
rf.fit(X_train, y_train)

# Prediksi
y_pred_rf = rf.predict(X_test)

# Accuracy
acc_rf = accuracy_score(y_test, y_pred_rf)

print("\n===== RANDOM FOREST =====")
print("Accuracy :", acc_rf)

# ==================================================
# CONFUSION MATRIX RANDOM FOREST
# ==================================================

cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(5,4))

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')

plt.title('Confusion Matrix Random Forest')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')

plt.show()

# ==================================================
# CLASSIFICATION REPORT RANDOM FOREST
# ==================================================

print("\nClassification Report Random Forest")
print(classification_report(y_test, y_pred_rf))

# ==================================================
# MODEL 2 : LOGISTIC REGRESSION
# ==================================================

lr = LogisticRegression(max_iter=1000)

# Training model
lr.fit(X_train, y_train)

# Prediksi
y_pred_lr = lr.predict(X_test)

# Accuracy
acc_lr = accuracy_score(y_test, y_pred_lr)

print("\n===== LOGISTIC REGRESSION =====")
print("Accuracy :", acc_lr)

# ==================================================
# CONFUSION MATRIX LOGISTIC REGRESSION
# ==================================================

cm_lr = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(5,4))

sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens')

plt.title('Confusion Matrix Logistic Regression')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')

plt.show()

# ==================================================
# CLASSIFICATION REPORT LOGISTIC REGRESSION
# ==================================================

print("\nClassification Report Logistic Regression")
print(classification_report(y_test, y_pred_lr))

# ==================================================
# PERBANDINGAN AKURASI
# ==================================================

print("\n===== PERBANDINGAN AKURASI =====")
print("Accuracy Random Forest       :", acc_rf)
print("Accuracy Logistic Regression :", acc_lr)

# ==================================================
# KESIMPULAN
# ==================================================

if acc_rf > acc_lr:
    print("\nKesimpulan:")
    print("Random Forest memiliki akurasi lebih tinggi dibandingkan Logistic Regression.")

elif acc_lr > acc_rf:
    print("\nKesimpulan:")
    print("Logistic Regression memiliki akurasi lebih tinggi dibandingkan Random Forest.")

else:
    print("\nKesimpulan:")
    print("Kedua model memiliki akurasi yang sama.")