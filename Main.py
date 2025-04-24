# %% [markdown]
# # 📊 Klasifikasi Kredit Komputer - Decision Tree

# %% 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# %% 
# 1. Load dataset
df = pd.read_csv('dataset_buys _comp.csv')  # Pastikan nama file sesuai

# %% 
# 2. Label Encoding untuk kolom kategorikal
df_encoded = df.copy()
label_encoders = {}

for column in df_encoded.columns:
    if df_encoded[column].dtype == 'object':
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df_encoded[column])
        label_encoders[column] = le

# %% 
# 3. Pisahkan fitur dan target
X = df_encoded.drop("Buys_Computer", axis=1)
y = df_encoded["Buys_Computer"]

# %% 
# 4. Split data menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Tambahan: Tampilkan jumlah data
print("=== Informasi Dataset ===")
print(f"Total data: {len(df)}")
print(f"Data latih: {len(X_train)}")
print(f"Data uji: {len(X_test)}\n")

# %% 
# 5. Buat dan latih model Decision Tree
model = DecisionTreeClassifier(random_state=42)  # Atur max_depth jika perlu
model.fit(X_train, y_train)

# %% 
# 6. Prediksi dan evaluasi model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Tidak", "Ya"])

# Tampilkan hasil evaluasi
print("=== Evaluasi Model Decision Tree ===")
print(f"Akurasi: {accuracy * 100:.2f}%\n")
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", report)

# %% 
# 7. Visualisasi pohon keputusan
plt.figure(figsize=(24, 12))
plot_tree(model,
          feature_names=X.columns,
          class_names=["Tidak", "Ya"],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree - Klasifikasi Kredit Komputer")
plt.show()
