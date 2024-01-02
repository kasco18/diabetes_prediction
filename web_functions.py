"""This module contains necessary function needed"""

# Import necessary modules
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier 
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split




#@st.cache_data
def load_data():
    # Load the Diabetes dataset into DataFrame.
    df = pd.read_csv('diabetes1.csv')

    
    # Check for duplicate values
    num_duplicates = df.duplicated().sum()
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Display the number of duplicates
    st.info(f"Number of Duplicated Rows that were Successfully dropped: {num_duplicates}")
    
    # Initialize LabelEncoder
    '''label_encoder = LabelEncoder()
    
    # Apply label encoding to categorical columns
    #df['gender'] = label_encoder.fit_transform(df['gender'])
    #df['smoking_history'] = label_encoder.fit_transform(df['smoking_history'])'''
    
    # Perform feature and target split
    X = df[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
    y = df["Outcome"]



    # Apply SMOTE method
    if st.checkbox("Apply SMOTE", key="smote"):
        # Your SMOTE code goes here
        # # Example: Assuming X and y are defined earlier
        smote = SMOTE(random_state=0)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Now you have resampled X_resampled and y_resampled
        # # You can proceed with your further analysis or modeling
        st.success("SMOTE Applied Successfully!")

        # Step 2: Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
        # The following line is optional. You can include it if you want to use the preprocessed data in other parts of your app.

        # Define the columns to remove outliers
        selected_columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
        # Calculate the IQR for the selected columns in the training data
        Q1 = X_train[selected_columns].quantile(0.25)
        Q3 = X_train[selected_columns].quantile(0.75)
        IQR = Q3 - Q1
        # SetTING a threshold value for outlier detection (e.g., 1.5 times the IQR)
        threshold = 1.5
        # CreatING a mask for outliers in the selected columns
        outlier_mask = (
            (X_train[selected_columns] < (Q1 - threshold * IQR)) |
            (X_train[selected_columns] > (Q3 + threshold * IQR))
            ).any(axis=1)
        # Remove rows with outliers from X_train and y_train
        X_train_clean = X_train[~outlier_mask]
        y_train_clean = y_train[~outlier_mask]
        # Print the number of rows removed
        num_rows_removed = len(X_train) - len(X_train_clean)
        st.info(f"Number of rows removed due to outliers: {num_rows_removed}")

        X, y = X_train_clean, y_train_clean
        return df, X, y
    return df, X, y


@st.cache_data
def train_dt(X, y):
    #This function trains the model and return the model and model score
    # Create the model
    model = DecisionTreeClassifier(
            ccp_alpha=0.0, class_weight=None, criterion='entropy',
            max_depth=4, max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_samples_leaf=1, 
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            random_state=42, splitter='best'
        )
    # Fit the data on model
    model.fit(X, y)
    # Get the model score
#    score = model.score(X, y)

    # Return the values
    return model


# Individual models
@st.cache_data  
def train_rf(X, y):
    # Define the parameter grid for RandomizedSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Create the RandomForestClassifier model
    model = RandomForestClassifier()

    # Create the RandomizedSearchCV model
    model = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=2, random_state=0)

    # Fit the data on the RandomizedSearchCV model
    model.fit(X, y)

    # Get the best model from the RandomizedSearchCV
    model = model.best_estimator_

    return model
   
@st.cache_data
def train_xgb(X, y):
    # Define a parameter grid for RandomizedSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'max_depth': [3, 5, 7, 10],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2, 0.3],
        'reg_alpha': [0, 0.1, 0.2, 0.3],
        'reg_lambda': [0, 0.1, 0.2, 0.3],
    }

    # Create the XGBClassifier model
    xgb_model = XGBClassifier()

    # Create the RandomizedSearchCV model
    model = RandomizedSearchCV(xgb_model, param_distributions=param_grid, n_iter=10, cv=5, random_state=0)

    # Fit the data on the RandomizedSearchCV model
    model.fit(X, y)

    # Get the best model from the RandomizedSearchCV
    model = model.best_estimator_

    return model


# Imports
@st.cache_data
def train_model(X, y):
    """Train an ensemble model"""
    
    # Individual models
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier() 
    xgb = XGBClassifier()

    # Ensemble
    voting_clf = VotingClassifier(
        estimators=[('dt', dt), ('rf', rf), ('xgb', xgb)],
        voting='hard'
    )

    # Train ensemble 
    voting_clf.fit(X, y)
    
    # Get score
    score = voting_clf.score(X, y)

    return voting_clf, score

# Rest of code remains same

def predict(X, y, features):
    # Get model and model score
    voting_clf, score = train_model(X, y)
    # Predict the value
    prediction = voting_clf.predict(np.array(features).reshape(1, -1))

    return prediction, score