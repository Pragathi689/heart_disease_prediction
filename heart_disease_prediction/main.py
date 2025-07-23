import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# 1. Load and preprocess dataset
df = pd.read_csv("Cardiovascular_Disease_Dataset.csv")
df.columns = df.columns.str.strip()
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Drop 'patientid' if it exists (not useful for prediction)
if 'patientid' in df.columns:
    df.drop(columns=['patientid'], inplace=True)

# 2. EDA
df = df[df['slope'].isin([0, 1, 2])]
os.makedirs("static", exist_ok=True)

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='gender', hue='target')
plt.title("Heart Disease by Gender")
plt.savefig("static/gender_heart_disease.png")
plt.close()

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("static/correlation_heatmap.png")
plt.close()

# 3. Train model
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate and Save model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_pred, y_test))

os.makedirs("model", exist_ok=True)
with open("model/heart_disease_rf_model.pkl", "wb") as f:
    pickle.dump(model, f)
