"""This modules contains data about home page"""

# Import necessary modules
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE



def app(df):
    """This function create the Data Info page"""
        # Perform feature and target split
    X = df[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
    y = df["Outcome"] 

    # Add title to the page
    st.title("Data Info page")

    # Add subheader for the section
    st.subheader("View Data")

    # Create an expansion option to check the data
    with st.expander("View data"):
        st.dataframe(df)

    # Create a section to columns values
    # Give subheader
    st.subheader("Columns Description:")

    # Create a checkbox to get the summary.
    if st.checkbox("View Summary"):
        st.dataframe(df.describe().transpose())

    if st.checkbox("Check Duplicates"):
        if len(y[y == 'Yes'].index) > 0:
            st.write('Duplicate entries in Yes column')
                # Check for duplicate values
        num_duplicates = df.duplicated().sum()
        # Drop duplicates
        df = df.drop_duplicates()
        # Display the number of duplicates
        st.info(f"Number of Drooped Duplicated Rows: {num_duplicates}")

    # Create multiple check box in row
    col_name, col_dtype, col_data = st.columns(3)

    # Show name of all dataframe
    with col_name:
        if st.checkbox("Column Names"):
            st.dataframe(df.columns)

    with col_data:
        if st.checkbox("Check Missing Value"):
            missing_data = df.isnull().sum()
            st.dataframe(missing_data)

    # Show datatype of all columns 
    with col_dtype:
        if st.checkbox("Columns data types"):
            dtypes = df.dtypes.apply(lambda x: x.name)
            st.dataframe(dtypes)
    
    # Show data for each columns
    with col_name: 
        if st.checkbox("Columns Data"):
            col = st.selectbox("Column Name", list(df.columns))
            st.dataframe(df[col])
            
    # Initialize LabelEncoder
    '''with col_dtype:
        if st.checkbox("Label Encoder of the dtypes"):
            label_encoder = LabelEncoder()
            # Apply label encoding to categorical columns
            ##df['gender'] = label_encoder.fit_transform(df['gender'])
            df['smoking_history'] = label_encoder.fit_transform(df['smoking_history'])
            # Display the updated DataFrame
            #st.dataframe(df)'''
    

# Show data for each column
    with col_data: 

    # Plot bar chart for 'diabetes' column
        if st.checkbox("Plot Diabetes Distribution"):
            st.bar_chart(df['Outcome'].value_counts())

    # Apply SMOTE method
        if st.checkbox("Apply SMOTE"):
        # Your SMOTE code goes here
        # Example: Assuming X and y are defined earlier
            

            smote = SMOTE(random_state=0)
            X_resampled, y_resampled = smote.fit_resample(X, y)

        # Now you have resampled X_resampled and y_resampled
        # You can proceed with your further analysis or modeling
            
                # Value counts after SMOTE
            st.write("Class distribution of Diabetes after SMOTE:") 
            st.bar_chart(y_resampled.value_counts())
            st.success("SMOTE Applied Successfully!")

    # Add the link to you dataset
    st.markdown("""
                    <p style="font-size:24px">
                        <a 
                            href="https://www.kaggle.com/uciml/pima-indians-diabetes-database"
                            target=_blank
                            style="text-decoration:none;"
                        >Get Dataset
                        </a> 
                    </p>
                """, unsafe_allow_html=True
    )