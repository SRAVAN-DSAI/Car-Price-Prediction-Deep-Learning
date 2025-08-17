import gradio as gr
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import datetime

# 1. Define the model architecture exactly as in your training script
class CarPricePredictor(nn.Module):
    def __init__(self, input_size):
        super(CarPricePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 2. Define a function to load the model and scaler
def load_assets():
    # Load the original dataset to recreate the scaler. 
    # This is a critical step to ensure the data is scaled correctly for new predictions.
    df = pd.read_csv("cardekho_data.csv")
    
    # Replicate preprocessing from ANN.py
    year = datetime.datetime.today().year
    df['Age'] = year - df['Year']
    df.drop('Year', axis=1, inplace=True)
    
    # Hardcode label encoder mappings for deployment
    # These values must be consistent with the values from your training data.
    car_name_map = {name: i for i, name in enumerate(df['Car_Name'].unique())}
    fuel_type_map = {name: i for i, name in enumerate(df['Fuel_Type'].unique())}
    seller_type_map = {name: i for i, name in enumerate(df['Seller_Type'].unique())}
    transmission_map = {name: i for i, name in enumerate(df['Transmission'].unique())}

    # Apply the mappings to create the feature set for the scaler
    df['Car_Name'] = df['Car_Name'].map(car_name_map)
    df['Fuel_Type'] = df['Fuel_Type'].map(fuel_type_map)
    df['Seller_Type'] = df['Seller_Type'].map(seller_type_map)
    df['Transmission'] = df['Transmission'].map(transmission_map)
    
    X = df.drop('Selling_Price', axis=1)
    
    # Create and fit the scaler on the preprocessed training features
    scaler = StandardScaler()
    scaler.fit(X.values)
    
    # Instantiate and load the trained model
    input_size = X.shape[1]
    model = CarPricePredictor(input_size)
    model.load_state_dict(torch.load('car_price_predictor.pth'))
    model.eval() # Set the model to evaluation mode
    
    return model, scaler, car_name_map, fuel_type_map, seller_type_map, transmission_map

# Load assets outside of the prediction function for efficiency
model, scaler, car_name_map, fuel_type_map, seller_type_map, transmission_map = load_assets()

# 3. Define the prediction function for Gradio
def predict_price(car_name, year, present_price, kms_driven, fuel_type, seller_type, transmission, owner):
    # Map the user input strings to their integer-encoded values
    car_name_le = car_name_map.get(car_name, -1) # Use .get() with a default value to handle new names
    fuel_type_le = fuel_type_map.get(fuel_type, -1)
    seller_type_le = seller_type_map.get(seller_type, -1)
    transmission_le = transmission_map.get(transmission, -1)

    if -1 in [car_name_le, fuel_type_le, seller_type_le, transmission_le]:
        return "Error: Invalid input for one or more categories."

    # Calculate the car's age
    age = datetime.datetime.today().year - year
    
    # Create a NumPy array with the input features in the correct order
    input_data = np.array([[
        car_name_le,
        present_price,
        kms_driven,
        fuel_type_le,
        seller_type_le,
        transmission_le,
        owner,
        age
    ]])

    # Standardize the input data using the trained scaler
    scaled_data = scaler.transform(input_data)
    
    # Convert to a PyTorch tensor
    tensor_data = torch.FloatTensor(scaled_data)
    
    # Make the prediction
    with torch.no_grad():
        prediction = model(tensor_data).item()
        
    return f"Predicted Price: {prediction:.2f} Lakhs"

# 4. Create the Gradio interface
gr_inputs = [
    gr.Dropdown(list(car_name_map.keys()), label="Car Name"),
    gr.Number(label="Year", value=2020),
    gr.Number(label="Present Price (Lakhs)", value=5.0),
    gr.Number(label="Kms Driven", value=50000),
    gr.Dropdown(list(fuel_type_map.keys()), label="Fuel Type"),
    gr.Dropdown(list(seller_type_map.keys()), label="Seller Type"),
    gr.Dropdown(list(transmission_map.keys()), label="Transmission"),
    gr.Number(label="Owner", value=1)
]

gr_outputs = gr.Textbox(label="Predicted Selling Price")

demo = gr.Interface(
    fn=predict_price,
    inputs=gr_inputs,
    outputs=gr_outputs,
    title="Car Price Prediction with Deep Learning",
    description="This application predicts the selling price of a car using a trained Neural Network model."
)

demo.launch()
