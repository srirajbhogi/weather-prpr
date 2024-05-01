import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load the model from the .h5 file
model = load_model('model_bnn.h5')

# Define the input dimension
input_dim = 7

# Take user input for features
user_input = []
feature_names = ["precipitation", "temp_max", "temp_min", "wind", "year", "month", "day"]
for feature_name in feature_names:
    feature_value = float(input(f"Enter value for {feature_name}: "))
    user_input.append(feature_value)

# Take only the first four features
user_input_array = np.array(user_input[:4]).reshape(1, -1)  # Reshape to a single sample

# Make prediction
output = model.predict(user_input_array)
predicted_class = np.argmax(output, axis=1)[0]

# Map predicted class to weather condition
weather_mapping = {0: 'drizzle', 1: 'fog', 2: 'rain', 3: 'snow', 4: 'sun'}
predicted_weather = weather_mapping.get(predicted_class, 'Unknown')

print(f"The predicted weather condition is: {predicted_weather}")
