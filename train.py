import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Path to your dataset
data_path = os.path.join("data", "train.csv")  # change if file name is different

# Load dataset
df = pd.read_csv(data_path)

# Select relevant columns
df = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Survived"]]

# Convert categorical to numeric
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Drop rows with missing values
df = df.dropna()

# Features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save model in the same folder as train.py
model_filename = "model.pkl"
with open(model_filename, "wb") as file:
    pickle.dump(model, file)

print(f"✅ Model trained and saved as {model_filename}")
