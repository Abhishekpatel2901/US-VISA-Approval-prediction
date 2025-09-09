import streamlit as st
import pickle
import pandas as pd
import gzip

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="US Visa Approval Predictor",
    page_icon="✅"
)

# --- Load Models and Preprocessor ---
try:
    with gzip.open('random_forest_model_compressed.pkl.gz', 'rb') as f:
        rf_model = pickle.load(f)
except FileNotFoundError:
    st.error("The model file ('random_forest_model_compressed.pkl.gz') was not found. Please ensure it is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

try:
    with open('PII_model.pickle', 'rb') as file:
        preprocessor = pickle.load(file)
except FileNotFoundError:
    st.error("The preprocessor file ('PII_model.pickle') was not found. Please ensure it is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the preprocessor: {e}")
    st.stop()


# --- Streamlit Application UI ---
st.title('US Visa Approval Prediction 🇺🇸')

st.markdown("""
Welcome to the Visa Approval Prediction App. This tool uses a machine learning model to predict the outcome of a visa application based on the details provided.
Please fill out the form below and click 'Predict Approval' to see the result.
""")

# Create columns for a more organized layout
col1, col2 = st.columns(2)

# Input fields for user data
with col1:
    st.subheader("Applicant & Employment Details")
    continent = st.selectbox(
        'Continent of Origin',
        ("Asia", "Europe", "North America", "South America", "Africa", "Oceania")
    )
    education_of_employee = st.selectbox(
        'Applicant\'s Education Level',
        ("High School", "Bachelor's", "Master's", "Doctorate")
    )
    has_job_experience = st.selectbox(
        'Does the applicant have prior job experience?',
        ('Y', 'N'),
        help="Select 'Y' for Yes, 'N' for No"
    )
    full_time_position = st.selectbox(
        'Is this a full-time position?',
        ('Y', 'N'),
        help="Select 'Y' for Yes, 'N' for No"
    )

with col2:
    st.subheader("Company & Wage Information")
    no_of_employees = st.number_input(
        'Number of Employees at the Company',
        min_value=1,
        value=500,
        step=10
    )
    company_age = st.number_input(
        'Company Age (in years)',
        min_value=0,
        value=25,
        step=1
    )
    prevailing_wage = st.number_input(
        'Prevailing Wage',
        min_value=0,
        value=85000,
        step=1000
    )
    unit_of_wage = st.selectbox(
        'Unit of Wage',
        ('Year', 'Month', 'Week', 'Hour')
    )
    # STEP 1: ADDING THIS INPUT WIDGET BACK IN
    region_of_employment = st.selectbox(
        'Region of Employment',
        ['Island', 'Midwest', 'Northeast', 'South', 'West']
    )


# --- Prediction Logic ---
if st.button('Predict Approval', type="primary"):
    # Create a DataFrame from the user inputs.
    # The dictionary keys MUST EXACTLY MATCH the column names your model was trained on.
    
    input_data = {
        'continent': [continent],
        'education_of_employee': [education_of_employee],
        'has_job_experience': [has_job_experience],
        'full_time_position': [full_time_position],
        'no_of_employees': [no_of_employees],
        'company_age': [company_age],
        'prevailing_wage': [prevailing_wage],
        'unit_of_wage': [unit_of_wage],
        # STEP 2: ADDING THE DATA FROM THE WIDGET TO THE DICTIONARY
        'region_of_employment': [region_of_employment]
    }

    input_df = pd.DataFrame(input_data)

    st.write("---")
    st.subheader("Prediction Result")
    
    try:
        # Use the preprocessor to transform the data
        transformed_data = preprocessor.transform(input_df)

        # Make a prediction and get probabilities
        prediction = rf_model.predict(transformed_data)
        prediction_proba = rf_model.predict_proba(transformed_data)

        # Display the result
        if prediction[0] == 1:  # Assuming 1 means 'Approved'
            st.success('**Result: Visa Approved**')
            st.progress(prediction_proba[0][1])
            st.write(f"The model predicts with a **{prediction_proba[0][1]*100:.2f}%** confidence that the visa will be approved.")
        else:
            st.error('**Result: Visa Denied**')
            st.progress(prediction_proba[0][0])
            st.write(f"The model predicts with a **{prediction_proba[0][0]*100:.2f}%** confidence that the visa will be denied.")

    except Exception as e:
        st.error(f"An error occurred during the prediction process: {e}")
        st.error("Please ensure that the column names in the app match the ones used for training the model.")