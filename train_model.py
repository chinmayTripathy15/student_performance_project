import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Step 1: Load dataset
df = pd.read_csv("data.csv")

# Step 2: Select required feature columns
feature_cols = ["hours_studied", "sleep_hours", "attendance_percent", "previous_scores"]

X = df[feature_cols]          # Features (input)
y = df["exam_score"]          # Target (output)

# Step 3: Train the model
model = LinearRegression()
model.fit(X, y)

# Step 4: Save the trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model successfully trained and saved as model.pkl")