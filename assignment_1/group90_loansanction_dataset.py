import pandas as pd

# Set the option to display all columns
pd.set_option('display.max_columns', 50)
# Your provided Google Drive file ID
file_id = '1WDtHB5FALrnPBYgSKK1AfufQnhzspKtk'
url = f'https://drive.google.com/uc?export=download&id={file_id}'

# Read the CSV file directly into a pandas DataFrame
df = pd.read_csv(url)
print(df.count())
# Now, exclude non-numeric columns before computing the correlation matrix
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns


# 2.a Display the first two rows of the DataFrame
# print(df.head(2))
#
# # Import necessary libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Check for class imbalance in the 'Loan_Status' column
class_counts = df['Loan_Status'].value_counts(normalize=True) * 100

# Correlational analysis
corr_matrix = df[numeric_columns].corr()

# Plotting the class imbalance
plt.figure(figsize=(10, 5))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='coolwarm')
plt.title('Loan Status Class Distribution')
plt.ylabel('Percentage')
plt.xlabel('Loan Status')

# Plotting the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numeric Features')

# plt.show()

# Returning class counts for text analysis
class_counts


# Data Pre-processing and Cleaning

# Checking for missing values in the dataset
missing_values = df.isnull().sum()

# Identifying outliers in numerical features
# Using IQR (Interquartile Range) method to identify outliers for 'ApplicantIncome' and 'LoanAmount'
Q1 = df[['ApplicantIncome', 'LoanAmount']].quantile(0.25)
Q3 = df[['ApplicantIncome', 'LoanAmount']].quantile(0.75)
IQR = Q3 - Q1

# Defining outliers as those beyond 1.5 times the IQR from the Q1 and Q3
outliers = ((df[['ApplicantIncome', 'LoanAmount']] < (Q1 - 1.5 * IQR)) |
            (df[['ApplicantIncome', 'LoanAmount']] > (Q3 + 1.5 * IQR))).sum()

# Checking skewness in 'ApplicantIncome' and 'LoanAmount'
skewness = df[['ApplicantIncome', 'LoanAmount']].skew()

# Outputting initial analysis for missing values, outliers, and skewness
print(missing_values, outliers, skewness)

import numpy as np

# Imputing missing values for categorical features with mode
categorical_features = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
for feature in categorical_features:
    df[feature].fillna(df[feature].mode()[0], inplace=True)

# Imputing missing values for numerical features with median
numerical_features = ['LoanAmount', 'Loan_Amount_Term']
for feature in numerical_features:
    df[feature].fillna(df[feature].median(), inplace=True)
# Correcting the previous block error by defining np for log transformation
df['ApplicantIncome_Imputed'] = df['ApplicantIncome'].apply(lambda x: np.log(x+1))
df['LoanAmount_Imputed'] = df['LoanAmount'].apply(lambda x: np.log(x+1))

# Dropping the original columns to avoid redundancy
df.drop(['ApplicantIncome', 'LoanAmount'], axis=1, inplace=True)
missing_values = df.isnull().sum()

print("New Missing Values : ",missing_values)
# Rechecking skewness after transformations
new_skewness = df[['ApplicantIncome_Imputed', 'LoanAmount_Imputed']].skew()

print("New Skewness: ",new_skewness)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Feature Engineering
# Encoding categorical variables using LabelEncoder
label_encoders = {}
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area', 'Loan_Status']

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Preparing the data for model building
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)  # Features

print(X.head(2))
y = df['Loan_Status']  # Target
print(y.head(2))
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Displaying the shape of the train and test sets to confirm successful split
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initializing the models
logistic_model = LogisticRegression(max_iter=200)
decision_tree_model = DecisionTreeClassifier(random_state=42)

# Training the Logistic Regression model
logistic_model.fit(X_train, y_train)
# Predicting on the test set
y_pred_logistic = logistic_model.predict(X_test)

# Training the Decision Tree model
decision_tree_model.fit(X_train, y_train)
# Predicting on the test set
y_pred_decision_tree = decision_tree_model.predict(X_test)

# Evaluating the models
# Logistic Regression performance
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
confusion_matrix_logistic = confusion_matrix(y_test, y_pred_logistic)
classification_report_logistic = classification_report(y_test, y_pred_logistic)

# Decision Tree performance
accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
confusion_matrix_decision_tree = confusion_matrix(y_test, y_pred_decision_tree)
classification_report_decision_tree = classification_report(y_test, y_pred_decision_tree)

print("\n\n",accuracy_logistic, accuracy_decision_tree, classification_report_logistic, classification_report_decision_tree)


from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning for Logistic Regression
param_grid_logistic = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [100, 200, 300]
}

# Grid search for Logistic Regression
grid_logistic = GridSearchCV(LogisticRegression(), param_grid_logistic, cv=5, scoring='accuracy')
grid_logistic.fit(X_train, y_train)

# Hyperparameter tuning for Decision Tree
param_grid_decision_tree = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10]
}

# Grid search for Decision Tree
grid_decision_tree = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_decision_tree, cv=5, scoring='accuracy')
grid_decision_tree.fit(X_train, y_train)

# Best parameters and best score for Logistic Regression
best_params_logistic = grid_logistic.best_params_
best_score_logistic = grid_logistic.best_score_

# Best parameters and best score for Decision Tree
best_params_decision_tree = grid_decision_tree.best_params_
best_score_decision_tree = grid_decision_tree.best_score_

(best_params_logistic, best_score_logistic, best_params_decision_tree, best_score_decision_tree)
