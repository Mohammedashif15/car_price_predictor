import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# === Step 1: Load the Dataset ===
df = pd.read_csv("car.csv")
print("✅ Dataset loaded successfully!\n")

# === Step 2: Preprocessing ===
print("📊 Dataset Head:")
print(df.head())

print("\n🧹 Cleaning data...")

# Drop rows with missing values (if any)
df.dropna(inplace=True)

# Convert categorical columns to numerical
df = pd.get_dummies(df, drop_first=True)

print("\n🔍 Cleaned Data Columns:")
print(df.columns)

# === Step 3: Feature and Target Split ===
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# === Step 4: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 5: Model Training ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Step 6: Prediction & Evaluation ===
y_pred = model.predict(X_test)

print("\n📈 Model Performance:")
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# === Step 7: Predict Sample ===
sample = X_test.iloc[0:1]
predicted_price = model.predict(sample)[0]
print(f"\n🚗 Predicted selling price for sample car: ₹{predicted_price:.2f}")

# === Step 8: Plot Actual vs Predicted ===
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.grid()
plt.tight_layout()
plt.show()
