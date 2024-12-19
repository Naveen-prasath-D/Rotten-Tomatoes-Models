import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_data(url):
    data = pd.read_csv(url, encoding='ISO-8859-1')
    return data

# Raw URL of the CSV file from GitHub
file_url = "https://raw.githubusercontent.com/Naveen-prasath-D/Rotten-Tomatoes-Models/main/Rotten_Tomatoes_Movies3.csv"
data = load_data(file_url)

# Preview the first few rows of the data
display(data)

# Basic information
#print(data.info())

# Check for missing values
#print(data.isnull().sum())

# Preview the dataset
#print(data.head())

# Drop rows with missing target values
data = data.dropna(subset=['audience_rating'])

# Fill or drop missing feature values
numeric_cols = data.select_dtypes(include=['number']).columns  # Select numeric columns only
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# Separate features and target
X = data.drop(columns=['audience_rating'])
y = data['audience_rating']

# Identify categorical and numerical columns
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing for numerical and categorical data
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combine preprocessors in a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Ratings")
plt.show()