import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report)
import joblib

print("="*50)
print("RETRAINING MODEL")
print("="*50)

# Load data
df = pd.read_csv('parkinsons.csv')
X = df.drop(['name', 'status'], axis=1)
y = df['status']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cross validation first
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
cv_scores = cross_val_score(
    model, X_scaled, y, cv=5)

print(f"\n5-Fold Cross Validation:")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score*100:.2f}%")
print(f"\nMean: {cv_scores.mean()*100:.2f}%")
print(f"Std: {cv_scores.std()*100:.2f}%")

# Train final model
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
model.fit(X_train, y_train)
preds = model.predict(X_test)

print(f"\nTest Accuracy: "
      f"{accuracy_score(y_test, preds)*100:.2f}%")
print("\nDetailed Report:")
print(classification_report(
    y_test, preds,
    target_names=['Healthy', 'At-Risk']))

# Save
joblib.dump(model, 'voice_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Model retrained and saved!")