import os
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Charger le dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Entraîner le modèle
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Évaluer
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# 5. Sauvegarder le modèle
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/model.pkl")
print("Modèle sauvegardé dans artifacts/model.pkl")