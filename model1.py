import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout 
import tensorflow as tf

data = pd.read_csv("seattle-weather.csv")

le = LabelEncoder()
data['weather_label'] = le.fit_transform(data['weather'])

# Create a dictionary mapping encoded labels to original weather values
weather_dict = {label: value for label, value in zip(data['weather_label'], data['weather'])}

data = data.drop('weather', axis=1).set_index('date')

print(data.head())

x = data.drop(["weather_label"], axis= 1)
y = data["weather_label"]

x_train, x_test ,y_train ,y_test = train_test_split(x,y, test_size=0.2,random_state=2)

# Define Bayesian Neural Network Model
model = Sequential()
model.add(Dense(units=32, activation='relu', kernel_initializer='glorot_uniform'))  # First hidden layer with ReLU activation
model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
model.add(Dense(units=16, activation='relu', kernel_initializer='glorot_uniform'))  # Second hidden layer with ReLU activation
model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
model.add(Dense(units=y.nunique(), activation='softmax'))  # Output layer with softmax activation (one-hot encoded)

# Compile the model with appropriate optimizer and loss function for BNN
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the BNN model
model.fit(x_train, y_train, epochs=50, batch_size=32)  # Adjust epochs and batch size as needed

# Evaluate the model on test data
predictions = model.predict(x_test)  # Use predict_classes for categorical predictions
predictions = np.argmax(predictions, axis=1)
test_accuracy = accuracy_score(y_test, predictions)

print('Accuracy Score on Test Data : {:.2f}%'.format(test_accuracy * 100))

# Save the BNN model (consider using a different filename to avoid conflicts)
model.save('model_bnn.h5')


