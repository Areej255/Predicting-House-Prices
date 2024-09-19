# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
# import the linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


train_data = pd.read_csv("C:\\Users\\PMLS\\Downloads\\train (1).csv")
test_data = pd.read_csv("C:\\Users\\PMLS\\Downloads\\test (1).csv")
print(train_data.head())
# checking for missing values
missing_data = train_data.isnull().sum().sort_values(ascending=False)
print(missing_data)
print(train_data.info)


# Import a function to split the data into training and test sets.
# from sklearn.model_selection import train_test_split
X = train_data.drop(columns=['Id', 'SalePrice'])  # Features/X is the dataset with all features except the target (house price).
y = train_data['SalePrice']  # Target variable/y is the target column (house prices).

# Separate numeric and categorical columns
numeric_cols = train_data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = train_data.select_dtypes(include=['object']).columns
# Fill missing values for numeric columns with mean
train_data[numeric_cols] = train_data[numeric_cols].fillna(train_data[numeric_cols].mean())

# Fill missing values for categorical columns with mode
train_data[categorical_cols] = train_data[categorical_cols].fillna(train_data[categorical_cols].mode().iloc[0])


# Encode categorical variables using One-Hot Encoding
X = pd.get_dummies(X, drop_first=True)  # drop_first avoids dummy variable trap

# Verify that missing values are filled or not 
print(train_data.isnull().sum())

scaler = StandardScaler()
numeric_features = train_data.select_dtypes(include=['float64', 'int64']).columns
train_data[numeric_features] = scaler.fit_transform(train_data[numeric_features])

# creating new feature "age"
train_data["Age"] = train_data["YrSold"] - train_data["YearBuilt"]


# Drop rows with missing values
X = X.dropna()

# Make sure to drop corresponding rows in y
y = y[X.index]
# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_train,X_test,y_train,y_test)


# build a linear regression model
# from sklearn.linear_model import LinearRegression
model = LinearRegression()
# fit the model using training data(learn the relationship between house prices and features)
model.fit(X_train, y_train)  #fit() --> Trains the model by finding the best-fitting line based on the training data.
predictions = model.predict(X_test)  # Make predictions on the test set
LinearRegression()
# print(predictions) #print predictions 


# Model Evaluation
rmse = np.sqrt(mean_squared_error(y_test,predictions))
mae = mean_absolute_error(y_test,predictions)
r2 = r2_score(y_test,predictions)
print(f"Root Mean Sqrd Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R squared: {r2}")


# visualization
# import matplotlib.pyplot as plt
plt.scatter(y_test,predictions)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

