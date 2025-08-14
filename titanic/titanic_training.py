import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv(r"D:\New folder (2)\titanic.csv")
df = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
df.dropna(inplace=True)

le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

joblib.dump(rf, "titanic_model.pkl")
joblib.dump(le, "sex_encoder.pkl")

print("Model and encoder saved successfully!")
