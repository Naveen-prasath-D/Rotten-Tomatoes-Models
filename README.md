# Predicting Audience Ratings for Rotten Tomatoes Movies

This document outlines the steps to build a machine learning pipeline that predicts audience ratings for movies using the Rotten Tomatoes dataset.

---

## Step 1: Import Necessary Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
```

### Documentation:
- **pandas**: Used for data manipulation and analysis.
- **numpy**: Provides support for numerical computations.
- **matplotlib.pyplot**: Enables data visualization.
- **seaborn**: Offers statistical data visualization.
- **sklearn.model_selection.train_test_split**: Splits the dataset into training and testing sets.
- **sklearn.preprocessing.StandardScaler**: Standardizes numerical features by scaling them.
- **sklearn.preprocessing.OneHotEncoder**: Encodes categorical variables into numerical format.
- **sklearn.compose.ColumnTransformer**: Applies transformations to specific columns.
- **sklearn.pipeline.Pipeline**: Creates a streamlined pipeline for preprocessing and modeling.
- **sklearn.ensemble.RandomForestRegressor**: Implements a random forest algorithm for regression tasks.
- **sklearn.metrics.mean_squared_error / r2_score**: Evaluates model performance.

---

## Step 2: Load the Dataset
```python
def load_data(url):
    data = pd.read_csv(url, encoding='ISO-8859-1')
    return data

# Raw URL of the CSV file from GitHub
file_url = "https://raw.githubusercontent.com/Naveen-prasath-D/Rotten-Tomatoes-Models/main/Rotten_Tomatoes_Movies3.csv"
data = load_data(file_url)

# Preview the first few rows of the data
display(data)
```

### Documentation:
- **load_data(url)**: Reads a CSV file from the provided URL and returns it as a pandas DataFrame.
- **ISO-8859-1**: Encoding used to handle special characters in the dataset.

---

## Step 3: Explore the Dataset
### Basic Information
```python
print(data.info())
```

### Check for Missing Values
```python
print(data.isnull().sum())
```

### Preview the Dataset
```python
print(data.head())
```

### Documentation:
- **data.info()**: Displays information about the dataset, including column names, data types, and non-null counts.
- **data.isnull().sum()**: Identifies columns with missing values and their counts.
- **data.head()**: Displays the first five rows of the dataset.

---

## Step 4: Handle Missing Data
```python
# Drop rows with missing target values
data = data.dropna(subset=['audience_rating'])

# Fill or drop missing feature values
numeric_cols = data.select_dtypes(include=['number']).columns  # Select numeric columns only
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
```

### Documentation:
- **data.dropna(subset)**: Removes rows with missing values in the specified columns.
- **data.select_dtypes(include)**: Selects columns of a specific data type (e.g., numeric).
- **data.fillna()**: Fills missing values with a specified value (median in this case).

---

## Step 5: Perform Feature Selection
```python
# Separate features and target
X = data.drop(columns=['audience_rating'])
y = data['audience_rating']

# Identify categorical and numerical columns
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
```

### Documentation:
- **X**: Contains the features used for prediction by removing the target column.
- **y**: Represents the target variable (audience ratings).
- **select_dtypes(include)**: Identifies columns of specific data types for preprocessing.

---

## Step 6: Create a Preprocessing Pipeline
```python
# Preprocessing for numerical and categorical data
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combine preprocessors in a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
```

### Documentation:
- **StandardScaler**: Standardizes numerical data by removing the mean and scaling to unit variance.
- **OneHotEncoder(handle_unknown='ignore')**: Encodes categorical variables and ignores unknown categories during transformation.
- **ColumnTransformer**: Applies specified transformations to corresponding columns.

---

## Step 7: Build the Model Pipeline
```python
# Define the pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])
```

### Documentation:
- **Pipeline**: Combines preprocessing and model steps into a single object for streamlined execution.
- **RandomForestRegressor**: Implements a random forest algorithm to predict audience ratings.

---

## Step 8: Split the Dataset
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Documentation:
- **train_test_split**: Splits the dataset into training and testing sets.
- **test_size=0.2**: Reserves 20% of the data for testing.
- **random_state=42**: Ensures reproducibility of the split.

---

## Step 9: Train the Model
```python
model.fit(X_train, y_train)
```

### Documentation:
- **model.fit()**: Trains the model using the training dataset.

---

## Step 10: Validate the Model
### Predictions and Metrics
```python
# Predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")
```

### Documentation:
- **model.predict()**: Generates predictions for the testing dataset.
- **mean_squared_error**: Computes the mean squared error between actual and predicted values.
- **r2_score**: Measures the proportion of variance explained by the model.

---

## Step 11: Visualize Results
```python
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Ratings")
plt.show()
```

### Documentation:
- **plt.scatter**: Creates a scatter plot to compare actual and predicted ratings.
- **plt.xlabel / plt.ylabel / plt.title**: Sets labels and titles for the plot.
- **plt.show()**: Displays the plot.

---

### Final Notes
This pipeline demonstrates how to preprocess data, build a machine learning model, and evaluate its performance effectively using Python and popular libraries.
