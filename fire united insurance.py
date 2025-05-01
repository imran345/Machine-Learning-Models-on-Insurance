#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# 1. Input data manually
data_dict = {
    'Period': ['Sep 2014 - Aug 2015', 'Sep 2014 - Aug 2015', 'Sep 2015 - Jun 2016', 'Sep 2015 - Jun 2016',
               'Jul 2016 - Jun 2017', 'Jul 2016 - Jun 2017', 'Jul 2017 - Jun 2018', 'Jul 2017 - Jun 2018',
               'Jul 2018 - Jun 2019', 'Jul 2018 - Jun 2019', 'Jul 2019 - Jun 2020', 'Jul 2019 - Jun 2020',
               'Jul 2020 - Jun 2021', 'Jul 2020 - Jun 2021', 'Jul 2021 - Jun 2022', 'Jul 2021 - Jun 2022',
               'Jul 2022 - Mar 2023', 'Jul 2022 - Mar 2023'],
    'Treaty Type': ['Quota Share', '1st Surplus'] * 9,
    'Contribution': [9117769, 14494140, 14625614, 40058668, 13215455, 44156668, 17436180, 63647322,
                     15214887, 45415621, 49668119, 12250589, 35440352, 8005019, 42873091, 14769544,
                     24265332, 11716474],
    'Rebate': [1641198, 2608945, 2632611, 7210560, 2378782, 7948200, 2664465, 7364613, 2738680, 8174812,
               7946899, 1960094, 5670456, 1280803, 6859695, 2363127, 3882453, 1874636],
    'Paid Losses': [3537375, 7877921, 4690866, 28811280, 11460347, 55789226, 13691857, 85972350, 3613279,
                    22828848, 29313313, 28179923, 16240926, 4487660, 28783304, 1069548, 853986, 6764],
    'Outstanding Losses': [0, 0, 0, 0, 114800, 459201, 6448, 61105, 491037, 11429136, 2436396, 1279189,
                           6546306, 467243, 100584971, 130026469, 586373, 187500],
    'Incurred Losses': [3537375, 7877921, 4690866, 28811280, 11575147, 56248427, 13698305, 86033455, 4104316,
                        34257984, 31749709, 29459112, 22787232, 4954903, 129368275, 131096017, 1440359, 194264],
    'Loss Ratio (%)': [39, 54, 32, 72, 88, 127, 79, 135, 27, 75, 64, 240, 64, 62, 302, 888, 6, 2],
    'Profit Balance': [3939196, 4007274, 7302137, 4036828, -738474, -20039959, 1073410, -29750746,
                       8371891, 2982825, 9971511, -19168617, 6982664, 1769313, -93354879, -118689600,
                       18942520, 9647574],
    'Net Profit': [3939196, 4007274, 7302137, 4036828, -738474, -20039959, 1073410, -29750746,
                   8371891, 2982825, 9971511, -19168617, 6982664, 1769313, -93354879, -118689600,
                   18942520, 9647574],
    'Profit Ratio (%)': [43, 28, 50, 10, -6, -45, 6, -47, 55, 7, 20, -156, 20, 22, -218, -804, 78, 82]
}
data = pd.DataFrame(data_dict)

# 2. Drop 'Period' column
data = data.drop(columns=['Period'])

# 3. Identify numeric and categorical columns
numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

# 4. Handle missing values in numeric columns with mean imputation
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# 5. Handle categorical columns
if len(categorical_cols) > 0:
    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

# 6. Multicollinearity removal
corr_matrix = data[numeric_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
data = data.drop(columns=to_drop)
numeric_cols = [col for col in numeric_cols if col not in to_drop]

# 7. Apply log1p (handle negative/zero values)
data[numeric_cols] = data[numeric_cols].applymap(lambda x: np.log1p(x) if x > -1 else np.nan)
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# 8. Split into features and target
target_column = numeric_cols[0]
X = data.drop(columns=[target_column])
y = data[target_column]

# 9. Normalize
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 10. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 11. Batch Gradient Descent
class BatchGradientDescentRegressor:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.losses = []

    def fit(self, X, y):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta = np.random.randn(X.shape[1])
        for epoch in range(self.epochs):
            gradients = -2/X.shape[0] * X.T.dot(y - X.dot(self.theta))
            self.theta -= self.learning_rate * gradients
            loss = np.mean((y - X.dot(self.theta))**2)
            self.losses.append(loss)

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return X.dot(self.theta)

model = BatchGradientDescentRegressor()
model.fit(X_train.values, y_train.values)
y_pred_gd = model.predict(X_test.values)

# 12. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 13. Evaluation
print("\nBatch Gradient Descent:")
print("MSE:", mean_squared_error(y_test, y_pred_gd))
print("R² Score:", r2_score(y_test, y_pred_gd))

print("\nLinear Regression:")
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R² Score:", r2_score(y_test, y_pred_lr))

# 14. Loss curve
plt.figure(figsize=(8, 5))
plt.plot(range(model.epochs), model.losses)
plt.title("Batch Gradient Descent Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)
plt.show()

# 15. Model Parameters
print("\nBatch GD Theta:", model.theta)
print("Linear Regression Coefficients:", lr.coef_)
print("Linear Regression Intercept:", lr.intercept_)


# In[4]:


# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# 1. Input data manually
data_dict = {
    'Period': ['Sep 2014 - Aug 2015', 'Sep 2014 - Aug 2015', 'Sep 2015 - Jun 2016', 'Sep 2015 - Jun 2016',
               'Jul 2016 - Jun 2017', 'Jul 2016 - Jun 2017', 'Jul 2017 - Jun 2018', 'Jul 2017 - Jun 2018',
               'Jul 2018 - Jun 2019', 'Jul 2018 - Jun 2019', 'Jul 2019 - Jun 2020', 'Jul 2019 - Jun 2020',
               'Jul 2020 - Jun 2021', 'Jul 2020 - Jun 2021', 'Jul 2021 - Jun 2022', 'Jul 2021 - Jun 2022',
               'Jul 2022 - Mar 2023', 'Jul 2022 - Mar 2023'],
    'Treaty Type': ['Quota Share', '1st Surplus'] * 9,
    'Contribution': [9117769, 14494140, 14625614, 40058668, 13215455, 44156668, 17436180, 63647322,
                     15214887, 45415621, 49668119, 12250589, 35440352, 8005019, 42873091, 14769544,
                     24265332, 11716474],
    'Rebate': [1641198, 2608945, 2632611, 7210560, 2378782, 7948200, 2664465, 7364613, 2738680, 8174812,
               7946899, 1960094, 5670456, 1280803, 6859695, 2363127, 3882453, 1874636],
    'Paid Losses': [3537375, 7877921, 4690866, 28811280, 11460347, 55789226, 13691857, 85972350, 3613279,
                    22828848, 29313313, 28179923, 16240926, 4487660, 28783304, 1069548, 853986, 6764],
    'Outstanding Losses': [0, 0, 0, 0, 114800, 459201, 6448, 61105, 491037, 11429136, 2436396, 1279189,
                           6546306, 467243, 100584971, 130026469, 586373, 187500],
    'Incurred Losses': [3537375, 7877921, 4690866, 28811280, 11575147, 56248427, 13698305, 86033455, 4104316,
                        34257984, 31749709, 29459112, 22787232, 4954903, 129368275, 131096017, 1440359, 194264],
    'Loss Ratio (%)': [39, 54, 32, 72, 88, 127, 79, 135, 27, 75, 64, 240, 64, 62, 302, 888, 6, 2],
    'Profit Balance': [3939196, 4007274, 7302137, 4036828, -738474, -20039959, 1073410, -29750746,
                       8371891, 2982825, 9971511, -19168617, 6982664, 1769313, -93354879, -118689600,
                       18942520, 9647574],
    'Net Profit': [3939196, 4007274, 7302137, 4036828, -738474, -20039959, 1073410, -29750746,
                   8371891, 2982825, 9971511, -19168617, 6982664, 1769313, -93354879, -118689600,
                   18942520, 9647574],
    'Profit Ratio (%)': [43, 28, 50, 10, -6, -45, 6, -47, 55, 7, 20, -156, 20, 22, -218, -804, 78, 82]
}
data = pd.DataFrame(data_dict)
print(data)


# In[10]:


# Identify numeric and categorical columns
numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

# Show the output
print("Numeric Columns:")
print(numeric_cols)

print("\nCategorical Columns:")
print(categorical_cols)


# In[12]:


# 4. Handle missing values in numeric columns with mean imputation
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
print(data[numeric_cols])



# In[13]:


# 5. Handle categorical columns
if len(categorical_cols) > 0:
    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

# Show the output (transformed DataFrame or categorical columns)
print("Transformed DataFrame:")
print(data.head())  # Display the first few rows to check the changes

print("\nTransformed Categorical Columns:")
print(categorical_cols)  # Display the list of categorical columns



# In[15]:


import numpy as np

# 6. Multicollinearity removal
corr_matrix = data[numeric_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
data = data.drop(columns=to_drop)
numeric_cols = [col for col in numeric_cols if col not in to_drop]

# Show the output of multicollinearity removal
print("Columns dropped due to multicollinearity:")
print(to_drop)

# Show the DataFrame after dropping highly correlated columns
print("\nData after multicollinearity removal:")
print(data.head())  # Display the first few rows of the data after dropping columns

# 7. Apply log1p (handle negative/zero values)
data[numeric_cols] = data[numeric_cols].applymap(lambda x: np.log1p(x) if x > -1 else np.nan)
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Show the output after applying log1p and filling missing values
print("\nData after applying log1p and filling missing values:")
print(data.head())  # Display the first few rows of the data after transformation


# In[17]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 8. Split into features and target
target_column = numeric_cols[0]
X = data.drop(columns=[target_column])
y = data[target_column]

# Show the output after splitting into features and target
print("Features (X) and Target (y) split:")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# 9. Normalize
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Show the output after normalization
print("\nNormalized Features (X_scaled) preview:")
print(X_scaled.head())  # Display the first few rows of normalized features

# 10. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Show the output after train-test split
print("\nTrain-Test Split:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


# In[20]:


import numpy as np

# 11. Batch Gradient Descent
class BatchGradientDescentRegressor:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.losses = []

    def fit(self, X, y):
        X = np.c_[np.ones((X.shape[0], 1)), X]  # Add intercept term (bias)
        self.theta = np.random.randn(X.shape[1])  # Initialize theta randomly
        for epoch in range(self.epochs):
            gradients = -2/X.shape[0] * X.T.dot(y - X.dot(self.theta))
            self.theta -= self.learning_rate * gradients  # Update theta
            loss = np.mean((y - X.dot(self.theta))**2)  # Compute the loss
            self.losses.append(loss)  # Store the loss

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]  # Add intercept term (bias)
        return X.dot(self.theta)  # Predict using the model

# Train the model
model = BatchGradientDescentRegressor()
model.fit(X_train.values, y_train.values)

# Make predictions
y_pred_gd = model.predict(X_test.values)

# Show the output

# Display the loss over epochs
print("Loss over epochs:")
print(model.losses[:10])  # Show the first 10 loss values to avoid printing too many

# Display the final model parameters (theta)
print("\nFinal model parameters (theta):")
print(model.theta)

# Display a preview of predicted vs actual values
print("\nPredicted vs Actual values (for the first 5 test samples):")
comparison_df = pd.DataFrame({'Actual': y_test.values[:5], 'Predicted': y_pred_gd[:5]})
print(comparison_df)


# In[21]:


import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 12. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 13. Evaluation
print("\nBatch Gradient Descent:")
print("MSE:", mean_squared_error(y_test, y_pred_gd))
print("R² Score:", r2_score(y_test, y_pred_gd))

print("\nLinear Regression:")
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R² Score:", r2_score(y_test, y_pred_lr))

# 14. Loss curve
plt.figure(figsize=(8, 5))
plt.plot(range(model.epochs), model.losses)
plt.title("Batch Gradient Descent Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)
plt.show()

# 15. Model Parameters
print("\nBatch GD Theta:", model.theta)
print("Linear Regression Coefficients:", lr.coef_)
print("Linear Regression Intercept:", lr.intercept_)


# In[ ]:




