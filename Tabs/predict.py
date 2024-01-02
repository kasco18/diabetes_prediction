"""This modules contains data about prediction page"""

# Import necessary modules
import streamlit as st
import numpy as np

# Import necessary functions from web_functions
from web_functions import predict, train_dt, train_rf, train_xgb


def app(df, X, y):
    """This function create the prediction page"""

    # Add title to the page
    st.title("Prediction Page")

    # Add a brief description
    st.markdown(
        """
            <p style="font-size:25px">
                This app uses <b style="color:green">Ensemble Learning Classifier</b> for the Prediction of Diabetes.
            </p>
        """, unsafe_allow_html=True)
    
    # Take feature input from the user
    # Add a subheader
    st.subheader("Select Values:")

    # Take input of features from the user.

     # Assuming df is your DataFrame
    df_description = {
        "Age": "Age in years",
        "BMI": "Body Mass Index (BMI)",
        "BloodPressure": "Blood Pressure",
        "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
        "Glucose": "Glucose Level",
        "Insulin": "Insulin Level",
        "Pregnancies": "Number of Pregnancies",
        "SkinThickness": "Skin Thickness"
        }

        # Take input of features from the user.
    # Take input of features from the user.
    Pregnancies = st.slider("Pregnancies", int(X["Pregnancies"].min()), int(X["Pregnancies"].max()), help=df_description["Pregnancies"])
    Glucose = st.slider("Glucose", int(X["Glucose"].min()), int(X["Glucose"].max()), help=df_description["Glucose"])
    BloodPressure = st.slider("BloodPressure", int(X["BloodPressure"].min()), int(X["BloodPressure"].max()), help=df_description["BloodPressure"])
    SkinThickness = st.slider("SkinThickness", int(X["SkinThickness"].min()), int(X["SkinThickness"].max()), help=df_description["SkinThickness"])
    Insulin = st.slider("Insulin", int(X["Insulin"].min()), int(X["Insulin"].max()), help=df_description["Insulin"])
    BMI = st.slider("BMI", float(X["BMI"].min()), float(X["BMI"].max()), help=df_description["BMI"])
    DiabetesPedigreeFunction = st.slider("DiabetesPedigreeFunction", float(X["DiabetesPedigreeFunction"].min()), float(X["DiabetesPedigreeFunction"].max()), help=df_description["DiabetesPedigreeFunction"])
    Age = st.slider("Age", int(X["Age"].min()), int(X["Age"].max()), help=df_description["Age"])


    

    # Get models
    dt_model = train_dt(X, y) 
    rf_model = train_rf(X, y)
    xgb_model = train_xgb(X, y)


    # Create a list to store all the features
    #features = [fg, ag, bp, sth, insulin, bmi, gc, age]

    features = [
        Age,
        BMI,
        BloodPressure,
        DiabetesPedigreeFunction,
        Glucose,
        Insulin,
        Pregnancies,
        SkinThickness]


    # Convert to NumPy array
    featuresa = np.array(features).reshape(1, -1)


    # Make predictions
    dt_pred = dt_model.predict([features])[0]
    rf_pred = rf_model.predict([features])[0]
    xgb_pred = xgb_model.predict(featuresa)[0]
    #ensemble_pred = ensemble_model.predict([features])


    # Create a button to predict
    if st.button("Predict"):
        # Get prediction and model score
        prediction, score = predict(X, y, features)        
        st.info("Predicted Sucessfully")

        # Print the output according to the prediction
        if (prediction == 1):
            st.warning("The person either has high risk of diabetes mellitus")
        else:
            st.success("The person is free from diabetes")

        # Display results 
        #st.write("Decision Tree Prediction:", dt_pred)
        #st.write("Random Forest Prediction:", rf_pred)
        #st.write("XGBoost Prediction:", xgb_pred)
        #st.write("Ensemble Prediction:", ensemble_pred[0])


        # Print output message 
        if dt_pred == 1:
            st.warning("DT predicts high risk for diabetes")
        else:
            st.success("DT predicts low risk for diabetes")
        
        if rf_pred == 1:
            st.warning("RF predicts high risk for diabetes")  
        else:
            st.success("RF predicts low risk for diabetes")
        
        if xgb_pred == 1:
            st.warning("XGB predicts high risk for diabetes")
        else:
            st.success("XGB predicts low risk for diabetes")


        dt_acc = dt_model.score(X, y)*100
        rf_acc = rf_model.score(X, y)*100
        xgb_acc = xgb_model.score(X, y)*100

        st.write("The Ensemble Learning model can be trusted by doctor coz of an accuracy of ", (score*100),"%")
        st.write("DT Accuracy:", dt_acc) 
        st.write("RF Accuracy:", rf_acc)
        st.write("XGB Accuracy:", xgb_acc)