import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load datasets
file_path = "data.csv"
df = pd.read_csv(file_path)

# Handle missing values
df.dropna(inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['Gender', 'City', 'Membership Type']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
X = df.drop(columns=['Customer ID', 'Total Spend', 'Satisfaction Level'])
y = df['Total Spend']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model MSE: {mse:.2f}")

# Save model
joblib.dump(model, "ecom_linear_regression_model.pkl")
print("Model saved as ecom_linear_regression_model.pkl")
