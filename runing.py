import streamlit as st
import tensorflow as tf
import base64
from keras.models import load_model
from streamlit_option_menu import option_menu
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
background-image: url("https://images.unsplash.com/photo-1518803194621-27188ba362c9?q");
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
    page = option_menu("Main Menu", ["Home", "About", "Analysis", "Contact"], 
        icons=['house', 'people', 'book', 'envelope'], menu_icon="cast", default_index=0)


if page == "Home":

    # Take user input for features
    user_input = []
    feature_names = ["precipitation", "temp_max", "temp_min", "wind", "year", "month", "day"]
    for feature_name in feature_names:
        if feature_name in ["year", "month", "day"]:
            feature_value = st.number_input(f"Enter value for {feature_name}:", step=1, format="%d", value=None)
        else:
            feature_value = st.number_input(f"Enter value for {feature_name}:", value=None)
        user_input.append(feature_value)

    # Add a button to trigger prediction
    if st.button("Predict Weather", key="predict_button"):
        if any(value is None for value in user_input):
            st.warning("Please enter all values to predict the weather.")
        else:
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

elif page == "Analysis":

    st.title("Analysis")
    st.write("Perform additional analysis here")

    option = st.selectbox(
        "Select a plot type:",
        ("Bar Chart", "Line Chart", "Scatter Chart", "Pair Plot"),
        index = None,
        placeholder="Select contact method...",

    )
    st.write('You selected:', option)

    data = pd.read_csv("seattle-weather.csv")

    if option == "Bar Chart":
        st.write(
            "Bar Chart",
            "A bar chart is a type of chart that presents categorical data with rectangular bars with heights or lengths proportional to the values that they represent.",
            sep="\n",
        )
        selected_columns = st.multiselect("Select columns for bar chart", data.columns)
        if selected_columns:
            if st.button("Generate Bar Chart"):
                st.write("Generating Bar Chart...")
                st.bar_chart(data[selected_columns])

    elif option == "Line Chart":
        st.write(
            "Line Chart",
            "A line chart or line plot is a type of plot which displays information as a series of data points called 'markers' connected by straight line segments.",
            sep="\n",
        )
        selected_columns = st.multiselect("Select columns for line chart", data.columns)
        if selected_columns:
            if st.button("Generate Line Chart"):
                st.write("Generating Line Chart...")
                st.line_chart(data[selected_columns])

    elif option == "Scatter Chart":
        st.write(
            "Scatter Chart",
            "A scatter plot is a type of plot or mathematical diagram using Cartesian coordinates to display values for two variables for a set of data.",
            sep="\n",
        )
        selected_x_column = st.selectbox("Select x-axis column for scatter plot", data.columns)
        selected_y_column = st.selectbox("Select y-axis column for scatter plot", data.columns)
        
        if st.button("Generate Scatter Plot"):
            st.write("Generating Scatter Plot...")
            scatter_plot = sns.scatterplot(data=data, x=selected_x_column, y=selected_y_column)
            st.pyplot(scatter_plot.get_figure())

    elif option == "Pair Plot":
        st.write(
            "Pair Plot",
            "A pair plot is a graphical display of the pairwise relationships in a dataset, allowing us to see both distribution of single variables and relationships between two variables.",
            sep="\n",
        )
        selected_columns = st.multiselect("Select columns for pair plot", data.columns)
        if selected_columns:
            if st.button("Generate Pair Plot"):
                st.write("Generating Pair Plot...")
                pair_plot = sns.pairplot(data[selected_columns])
                st.pyplot(pair_plot.fig)

    

elif page == "Contact":

    st.title("Contact Us")
    st.write(":mailbox: Get In Touch With Me!")

    contact_form = """
    <form action="https://formsubmit.co/YOUREMAIL@EMAIL.COM" method="POST">
         <input type="hidden" name="_captcha" value="false">
         <input type="text" name="name" placeholder="Your name" required>
         <input type="email" name="email" placeholder="Your email" required>
         <textarea name="message" placeholder="Your message here"></textarea>
         <button type="submit">Send</button>
    </form>
    """

    st.markdown(contact_form, unsafe_allow_html=True)

    # Use Local CSS File
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    local_css("style.css")

