import streamlit as st
import tensorflow as tf
import base64
from keras.models import load_model
from streamlit_option_menu import option_menu
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Weather Prediction App", page_icon=":partly_sunny:")

df = px.data.iris()

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("templates/drizzle_image.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://images.unsplash.com/photo-1501426026826-31c667bdf23d");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("Weather Prediction App")

# Load the model from the .h5 file
model = load_model('model_bnn.h5')

# Define the input dimension
input_dim = 7

# Initialize predicted weather condition and button state
predicted_weather = ''
button_clicked = False

# Sidebar navigation
with st.sidebar:
    page = option_menu("Main Menu", ["Home", 'About'], 
        icons=['house', 'gear'], menu_icon="cast", default_index=1)
    page

if page == "Home":

    # Take user input for features
    user_input = []
    feature_names = ["precipitation", "temp_max", "temp_min", "wind", "year", "month", "day"]
    for feature_name in feature_names:
        if feature_name in ["year", "month", "day"]:
            feature_value = st.number_input(f"Enter value for {feature_name}:", step=1, format="%d")
        else:
            feature_value = st.number_input(f"Enter value for {feature_name}:")
        user_input.append(feature_value)

    # Add a button to trigger prediction
    if st.button("Predict Weather", key="predict_button"):
        button_clicked = True

    # Make prediction if button is clicked
    if button_clicked:
        # Take only the first four features
        user_input_array = np.array(user_input[:4]).reshape(1, -1)  # Reshape to a single sample

        # Make prediction
        output = model.predict(user_input_array)
        predicted_class = np.argmax(output, axis=1)[0]

        # Map predicted class to weather condition
        weather_mapping = {0: 'drizzle', 1: 'fog', 2: 'rain', 3: 'snow', 4: 'sun'}
        predicted_weather = weather_mapping.get(predicted_class, 'Unknown')

    # Display the predicted weather condition in a box if prediction is made
    if predicted_weather:
        st.success(f"The predicted weather condition is: {predicted_weather}")

elif page == "About":

    st.markdown(
    """
    ## Bayesian Deep Learning Model

    This app uses a Bayesian deep learning model to predict weather conditions based on the input features. 
    Bayesian deep learning models provide a probabilistic approach to deep learning, allowing for uncertainty 
    estimation in predictions. In this app, the model takes into account various weather features such as 
    precipitation, temperature, wind, and date to predict the most likely weather condition.

    To make a prediction, enter the values for each feature and click the "Predict Weather" button.
    """
    )

    st.title("About")
    st.write(
        """
        ## Weather Prediction App

        This app uses a Bayesian deep learning model to predict weather conditions based on the input features.
        """
    )
    st.write(
        """
        ### How it works

        - Enter the values for each weather feature such as precipitation, temperature, wind, year, month, and day.
        - Click the "Predict Weather" button to see the predicted weather condition.
        - The app uses a Bayesian deep learning model to make the prediction.
        """
    )



