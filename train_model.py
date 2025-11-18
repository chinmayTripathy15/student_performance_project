import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Load the dataset
df = pd.read_csv('data.csv')

# Features and labels
X = df[['study_hours', 'sleep_hours', 'attendance']]
y = df['pass']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model to file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved as model.pkl")
