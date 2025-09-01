import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("[*] Loading dataset...")
# Load dataset (KDDCup 99+ has no header, so add generic ones)
columns = [i for i in range(42)]  # 41 features + 1 label
train_df = pd.read_csv("KDDTrain+.txt", names=columns)
test_df = pd.read_csv("KDDTest+.txt", names=columns)

print("[*] Encoding categorical features...")

for col in train_df.columns:
    if train_df[col].dtype == 'object':
        encoder = LabelEncoder()
        # Fit encoder on both train and test unique values
        combined_data = pd.concat([train_df[col], test_df[col]], axis=0)
        encoder.fit(combined_data)
        train_df[col] = encoder.transform(train_df[col])
        test_df[col] = encoder.transform(test_df[col])

X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

print("[*] Training the IDS model...")
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

print("[*] Evaluating the model...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"[+] Accuracy: {acc * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
print("[+] Confusion matrix saved as confusion_matrix.png")
