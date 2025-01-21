import warnings
from pickle import dump

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Load the dataset
print("Loading the dataset...")
df = pd.read_csv("insurance.csv")
print("Dataset loaded successfully!")

# Display the first few rows of the dataset
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Display information about the dataset
print("\nDataset Info:")
print(df.info())

# Display statistical summary of numerical columns
print("\nStatistical Summary:")
print(df.describe())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Pie charts for categorical features
features = ['sex', 'smoker', 'region']
plt.subplots(figsize=(20, 10))
print("\nCreating pie charts for categorical features...")
for i, col in enumerate(features):
    plt.subplot(1, 3, i + 1)
    x = df[col].value_counts()
    plt.pie(x.values, labels=x.index, autopct='%1.1f%%')
    plt.title(f"Distribution of {col}")
plt.show()

# Bar plots for mean charges grouped by categorical features
features = ['sex', 'children', 'smoker', 'region']
plt.subplots(figsize=(20, 10))
print("\nCreating bar plots for mean charges grouped by categorical features...")
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    df.groupby(col)['charges'].mean().astype(float).plot.bar()
    plt.title(f"Mean charges by {col}")
plt.show()

# Scatter plots for numerical features vs charges
features = ['age', 'bmi']
plt.subplots(figsize=(17, 7))
print("\nCreating scatter plots for numerical features vs charges...")
for i, col in enumerate(features):
    plt.subplot(1, 2, i + 1)
    sns.scatterplot(data=df, x=col, y='charges', hue='smoker')
    plt.title(f"{col} vs Charges")
plt.show()

# Boxplot for 'age'
print("\nCreating boxplot for age...")
sns.boxplot(df['age'])
plt.title("Boxplot of Age")
plt.show()

# Display the dataset shape before cleaning
print("\nDataset shape before cleaning:", df.shape)

# Boxplot for 'bmi'
print("\nCreating boxplot for bmi...")
sns.boxplot(df['bmi'])
plt.title("Boxplot of BMI")
plt.show()

# Calculate IQR for 'bmi' and define limits for outliers
Q1 = df['bmi'].quantile(0.25)
Q3 = df['bmi'].quantile(0.75)
iqr = Q3 - Q1
lowlim = Q1 - 1.5 * iqr
upplim = Q3 + 1.5 * iqr
print("\nBMI Outlier Limits:")
print(f"Lower Limit: {lowlim}")
print(f"Upper Limit: {upplim}")

# Remove outliers in 'bmi'
print("\nRemoving outliers in BMI...")
df['bmi'] = np.where(df['bmi'] > upplim, upplim, df['bmi'])
df['bmi'] = np.where(df['bmi'] < lowlim, lowlim, df['bmi'])

# Boxplot for 'bmi' after outlier removal
sns.boxplot(df['bmi'])
plt.title("Boxplot of BMI (after outlier removal)")
plt.show()

# Display skewness of 'bmi' and 'age'
print("\nSkewness:")
print(f"BMI Skewness: {df['bmi'].skew()}")
print(f"Age Skewness: {df['age'].skew()}")

# Convert categorical features to strings
print("\nConverting categorical features to strings...")
df['sex'] = df['sex'].astype(str)
df['smoker'] = df['smoker'].astype(str)
df['region'] = df['region'].astype(str)
print("Conversion complete!")

# Encode categorical features
print("\nEncoding categorical features...")
df['sex'] = df['sex'].replace({'male': 0, 'female': 1})
df['smoker'] = df['smoker'].replace({'yes': 1, 'no': 0})
df['region'] = df['region'].replace({'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3})

# Convert encoded features to float
print("\nConverting encoded features to float...")
df['sex'] = df['sex'].astype(np.float64)
df['smoker'] = df['smoker'].astype(np.float64)
df['region'] = df['region'].astype(np.float64)
print("Conversion complete!")

# Drop any remaining missing values
print("\nDropping any remaining missing values...")
df.dropna(inplace=True)

# Display updated dataset info
print("\nUpdated Dataset Info:")
print(df.info())

# Display dataset shape
print("\nDataset shape after cleaning:", df.shape)

# Correlation matrix
print("\nCorrelation Matrix:")
print(df.corr())

# Split features and target
X = df.drop(['charges'], axis=1)
Y = df[['charges']]

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Train-test split with Linear Regression analysis
print("\nPerforming Linear Regression with varying random states...")
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score

l1 = []  # Train accuracies
l2 = []  # Test accuracies
l3 = []  # Cross-validation scores

for i in range(40, 50):
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=i)
    lrmodel = LinearRegression()
    lrmodel.fit(xtrain, ytrain)
    l1.append(lrmodel.score(xtrain, ytrain))
    l2.append(lrmodel.score(xtest, ytest))
    cvs = cross_val_score(lrmodel, X, Y, cv=5).mean()
    l3.append(cvs)
    df1 = pd.DataFrame({'Train Accuracy': l1, 'Test Accuracy': l2, 'CV Score': l3})
    print(f"\nRandom State {i} Results:")
    print(df1)

# Final train-test split
print("\nPerforming final train-test split with random state 42...")
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
print("Shapes after split:")
print(f"X Train: {xtrain.shape}, Y Train: {ytrain.shape}")
print(f"X Test: {xtest.shape}, Y Test: {ytest.shape}")

# Flatten target variables if necessary
ytrain = ytrain.squeeze()
ytest = ytest.squeeze()
print("Shapes after flattening:")
print(f"Y Train: {ytrain.shape}, Y Test: {ytest.shape}")

# Linear Regression Model
print("\nTraining Linear Regression model...")
lrmodel = LinearRegression()
lrmodel.fit(xtrain, ytrain)
print("Train R2 Score:", lrmodel.score(xtrain, ytrain))
print("Test R2 Score:", lrmodel.score(xtest, ytest))
print("Cross-validation Mean Score:", cross_val_score(lrmodel, X, Y, cv=5).mean())

# SVR Model
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# Support Vector Regressor Model
print("Training Support Vector Regressor (SVR)...")
svrmodel = SVR()
svrmodel.fit(xtrain, ytrain)
ypredtrain1 = svrmodel.predict(xtrain)
ypredtest1 = svrmodel.predict(xtest)
print("R2 score on training data (SVR):", r2_score(ytrain, ypredtrain1))
print("R2 score on test data (SVR):", r2_score(ytest, ypredtest1))
print("Cross-validation score (SVR):", cross_val_score(svrmodel, X, Y, cv=5).mean())

# Random Forest Regressor Model
print("\nTraining Random Forest Regressor...")
rfmodel = RandomForestRegressor(random_state=42)
rfmodel.fit(xtrain, ytrain)
ypredtrain2 = rfmodel.predict(xtrain)
ypredtest2 = rfmodel.predict(xtest)
print("R2 score on training data (Random Forest):", r2_score(ytrain, ypredtrain2))
print("R2 score on test data (Random Forest):", r2_score(ytest, ypredtest2))
print("Cross-validation score (Random Forest):", cross_val_score(rfmodel, X, Y, cv=5).mean())

# Hyperparameter Tuning for Random Forest
print("\nPerforming Grid Search for Random Forest Regressor...")
estimator = RandomForestRegressor(random_state=42)
param_grid = {'n_estimators': [10, 40, 50, 98, 100, 120, 150]}
grid = GridSearchCV(estimator, param_grid, scoring="r2", cv=5)
grid.fit(xtrain, ytrain)
print("Best parameters for Random Forest:", grid.best_params_)
rfmodel = RandomForestRegressor(random_state=42, n_estimators=120)
rfmodel.fit(xtrain, ytrain)
ypredtrain2 = rfmodel.predict(xtrain)
ypredtest2 = rfmodel.predict(xtest)
print("R2 score on training data (Tuned Random Forest):", r2_score(ytrain, ypredtrain2))
print("R2 score on test data (Tuned Random Forest):", r2_score(ytest, ypredtest2))
print("Cross-validation score (Tuned Random Forest):", cross_val_score(rfmodel, X, Y, cv=5).mean())

# Gradient Boosting Regressor Model
print("\nTraining Gradient Boosting Regressor...")
gbmodel = GradientBoostingRegressor()
gbmodel.fit(xtrain, ytrain)
ypredtrain3 = gbmodel.predict(xtrain)
ypredtest3 = gbmodel.predict(xtest)
print("R2 score on training data (Gradient Boosting):", r2_score(ytrain, ypredtrain3))
print("R2 score on test data (Gradient Boosting):", r2_score(ytest, ypredtest3))
print("Cross-validation score (Gradient Boosting):", cross_val_score(gbmodel, X, Y, cv=5).mean())

# Hyperparameter Tuning for Gradient Boosting
print("\nPerforming Grid Search for Gradient Boosting Regressor...")
estimator = GradientBoostingRegressor()
param_grid = {'n_estimators': [10, 15, 19, 20, 21, 50], 'learning_rate': [0.1, 0.19, 0.2, 0.21, 0.8, 1]}
grid = GridSearchCV(estimator, param_grid, scoring="r2", cv=5)
grid.fit(xtrain, ytrain)
print("Best parameters for Gradient Boosting:", grid.best_params_)
gbmodel = GradientBoostingRegressor(n_estimators=19, learning_rate=0.2)
gbmodel.fit(xtrain, ytrain)
ypredtrain3 = gbmodel.predict(xtrain)
ypredtest3 = gbmodel.predict(xtest)
print("R2 score on training data (Tuned Gradient Boosting):", r2_score(ytrain, ypredtrain3))
print("R2 score on test data (Tuned Gradient Boosting):", r2_score(ytest, ypredtest3))
print("Cross-validation score (Tuned Gradient Boosting):", cross_val_score(gbmodel, X, Y, cv=5).mean())

# XGBoost Regressor Model
print("\nTraining XGBoost Regressor...")
xgmodel = XGBRegressor()
xgmodel.fit(xtrain, ytrain)
ypredtrain4 = xgmodel.predict(xtrain)
ypredtest4 = xgmodel.predict(xtest)
print("R2 score on training data (XGBoost):", r2_score(ytrain, ypredtrain4))
print("R2 score on test data (XGBoost):", r2_score(ytest, ypredtest4))
print("Cross-validation score (XGBoost):", cross_val_score(xgmodel, X, Y, cv=5).mean())

# Hyperparameter Tuning for XGBoost
print("\nPerforming Grid Search for XGBoost Regressor...")
estimator = XGBRegressor()
param_grid = {'n_estimators': [10, 15, 20, 40, 50], 'max_depth': [3, 4, 5], 'gamma': [0, 0.15, 0.3, 0.5, 1]}
grid = GridSearchCV(estimator, param_grid, scoring="r2", cv=5)
grid.fit(xtrain, ytrain)
print("Best parameters for XGBoost:", grid.best_params_)
xgmodel = XGBRegressor(n_estimators=15, max_depth=3, gamma=0)
xgmodel.fit(xtrain, ytrain)
ypredtrain4 = xgmodel.predict(xtrain)
ypredtest4 = xgmodel.predict(xtest)
print("R2 score on training data (Tuned XGBoost):", r2_score(ytrain, ypredtrain4))
print("R2 score on test data (Tuned XGBoost):", r2_score(ytest, ypredtest4))
print("Cross-validation score (Tuned XGBoost):", cross_val_score(xgmodel, X, Y, cv=5).mean())

# Feature Importance Analysis
print("\nAnalyzing Feature Importance...")
feats = pd.DataFrame(data=grid.best_estimator_.feature_importances_, index=X.columns, columns=['Importance'])
print("Importance Features:")
print(feats)

important_features = feats[feats['Importance'] > 0.01]
print("Important Features:")
print(important_features)

# Final Model Training
print("\nFinal Model Training with Important Features...")
# df.drop(df[['sex', 'region']], axis=1, inplace=True)
Xf = df.drop(df[['charges']], axis=1)
xtrain, xtest, ytrain, ytest = train_test_split(Xf, Y, test_size=0.2, random_state=42)
finalmodel = XGBRegressor(n_estimators=15, max_depth=3, gamma=0)  # HYPER TUNED VALUES used
finalmodel.fit(xtrain, ytrain)
ypredtrain4 = finalmodel.predict(xtrain)
ypredtest4 = finalmodel.predict(xtest)
print("R2 score on training data (Final Model):", r2_score(ytrain, ypredtrain4))
print("R2 score on test data (Final Model):", r2_score(ytest, ypredtest4))
print("Cross-validation score (Final Model):", cross_val_score(finalmodel, X, Y, cv=5).mean())
print("Execution Completed")

# SAVING MODEL

dump(finalmodel, open('Insurance_ML.pkl', 'wb'))

# NEW DATAFRAME TO PREDICT
new_data = pd.DataFrame({'age': 19, 'sex': 'male', 'bmi': 28, 'children': 0,
                         'smoker': 'yes', 'region': 'northeast'}, index=[0])
print("DATA INPUT IS: ", new_data.T)

# DATA MANIPULATION ACCORDING TO MODEL
"""
df['sex'] = df['sex'].replace({'male': 0, 'female': 1})
df['smoker'] = df['smoker'].replace({'yes': 1, 'no': 0})
df['region'] = df['region'].replace({'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3})
"""
new_data['smoker'] = new_data['smoker'].map({'yes': 1, 'no': 0})  # DATA ENCODING
new_data['sex'] = new_data['smoker'].map({'male': 0, 'female': 1})  # DATA ENCODING
new_data['region'] = new_data['region'].map({'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3})

# new_data = new_data.drop(new_data[['sex', 'region']], axis=1)  # USELESS FEATURES

# DATA PREDICTION
predicted = finalmodel.predict(new_data)
print("PREDICTED VALUE IS: ", predicted)
