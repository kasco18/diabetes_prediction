"""This modules contains data about visualisation page"""

# Import necessary modules
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import tree
import streamlit as st


# Import necessary functions from web_functions
from web_functions import train_model, train_dt

def app(df, X, y):
    """This function create the visualisation page"""
    
    # Remove the warnings
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Set the page title
    st.title("Visualise the Diabetes Prediction Web app")

    # Create a checkbox to show correlation heatmap
    if st.checkbox("Show the correlation heatmap"):
        st.subheader("Correlation Heatmap")

        fig = plt.figure(figsize = (10, 6))
        ax = sns.heatmap(df.iloc[:, 1:].corr(), annot = True)   # Creating an object of seaborn axis and storing it in 'ax' variable
        bottom, top = ax.get_ylim()                             # Getting the top and bottom margin limits.
        ax.set_ylim(bottom + 0.5, top - 0.5)                    # Increasing the bottom and decreasing the top margins respectively.
        st.pyplot(fig)    
    X = df[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
    y = df["Outcome"] 
    
    if st.checkbox("Preganacies vs Age Plot"):
        sns.color_palette("rocket", as_cmap=True)
        ax=sns.scatterplot(x="Pregnancies",y="Age",data=df)
        st.pyplot()

    if st.checkbox("BloodPressure vs BMI"):
        sns.color_palette("rocket", as_cmap=True)
        ax=sns.scatterplot(x="BMI",y="BloodPressure",data=df)
        st.pyplot()

    if st.checkbox("Glucose vs Insulin level"):
        sns.color_palette("rocket", as_cmap=True)
        ax=sns.scatterplot(x="Glucose",y="Insulin",data=df)
        st.pyplot()

    if st.checkbox("Show Histogram of Age against Diabetes Pedigree Functione"):
        sns.color_palette("rocket", as_cmap=True)
        ax=sns.histplot(df,x="Age",y="DiabetesPedigreeFunction")
        st.pyplot()

    if st.checkbox("AGe VS BMI"):
        sns.color_palette("rocket", as_cmap=True)
        ax=sns.histplot(df,x="Age",y="BMI")
        st.pyplot()    
    
    if st.checkbox("BMI VS DiabetesPedigreeFunction"):
        sns.color_palette("rocket", as_cmap=True)
        ax=sns.histplot(df,x="BMI",y="DiabetesPedigreeFunction")
        st.pyplot() 

    if st.checkbox("Plot Decision Tree"):
        model = train_dt(X, y)
        # Export decision tree in dot format and store in 'dot_data' variable.
        dot_data = tree.export_graphviz(
            decision_tree=model, max_depth=3, out_file=None, filled=True, rounded=True,
            feature_names=X.columns, class_names=['0', '1']
        )
        # Plot the decision tree using the 'graphviz_chart' function of the 'streamlit' module.
        st.graphviz_chart(dot_data)

        


    if hasattr(train_model, 'estimators'):
        # Assume you want to access the first base estimator
        base_estimator = train_model.estimators[0]
        if st.checkbox("Plot Decision Tree"):
            # Export decision tree in dot format and store in 'dot_data' variable.
            dot_data = tree.export_graphviz(
            decision_tree=base_estimator, max_depth=3, out_file=None, filled=True, rounded=True,
            featuresa_names=X.columns, class_names=['0', '1']
        )
            # Plot the decision tree using the 'graphviz_chart' function of the 'streamlit' module.
            st.graphviz_chart(dot_data)
    else:
        st.warning("The trained model doesn't have the 'estimators_' attribute.")

