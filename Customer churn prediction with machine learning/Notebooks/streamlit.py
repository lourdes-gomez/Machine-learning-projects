import streamlit as st
import pickle
import numpy as np

# Load the trained machine learning model
model_filename = r'C:\Users\lour2\Desktop\LOURDES\data science\Proyecto Machine Learning\my_model.pkl'  # Replace with your actual model filename
with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit App
st.title("Machine Learning Prediction App")

# Sidebar with user input
st.sidebar.header("User Input")

# Example: Dropdown for selecting a feature
feature_option = st.sidebar.selectbox("Select a Feature", ["Feature1", "Feature2", "Feature3"])

# Example: Text input for a numerical value
numerical_value = st.sidebar.number_input("Enter a numerical value")

# Button to make predictions
if st.sidebar.button("Make Prediction"):
    # Preprocess the input data (modify this based on your preprocessing steps)
    input_data = np.array([[numerical_value]])  # Example: create a 2D array for input data
    prediction = model.predict(input_data)

    # Display the prediction
    st.write(f"Prediction: {prediction}")

# Additional features and information can be added to the main section of the app
st.header("Additional Information")

# Add more content or visualizations as needed

# Run the app with: streamlit run your_app_filename.py