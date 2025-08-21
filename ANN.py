import gradio as gr
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import KFold, train_test_split
import pickle
import os
import datetime
import logging
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='app.log')

# Define the neural network model
class CarPricePredictor(nn.Module):
    def __init__(self, input_size):
        super(CarPricePredictor, self).__init__()
        self.layer1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.layer2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.layer3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.layer4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.08)  # Adjusted from 0.1
        self.layer5 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.layer1(x)))
        x = torch.relu(self.bn2(self.layer2(x)))
        x = torch.relu(self.bn3(self.layer3(x)))
        x = torch.relu(self.bn4(self.layer4(x)))
        x = self.dropout(x)
        x = self.layer5(x)
        return x

# Load and preprocess data with polynomial features
def load_and_preprocess_data():
    try:
        df = pd.read_csv("cardekho_data.csv")
        logging.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        logging.info(f"Dataset stats:\n{df[['Selling_Price', 'Present_Price']].describe()}")
        
        current_year = datetime.datetime.today().year  # 2025-08-21
        df['Age'] = current_year - df['Year']
        df.drop('Year', axis=1, inplace=True)
        
        for col in ['Present_Price', 'Kms_Driven', 'Age']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
        
        label_encoders = {}
        categorical_cols = ['Car_Name', 'Fuel_Type', 'Seller_Type', 'Transmission']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        poly = PolynomialFeatures(degree=2, include_bias=False)
        numeric_cols = ['Present_Price', 'Kms_Driven', 'Age']
        poly_features = poly.fit_transform(df[numeric_cols])
        poly_feature_names = poly.get_feature_names_out(numeric_cols)
        df_poly = pd.DataFrame(poly_features, columns=poly_feature_names)
        df = pd.concat([df, df_poly], axis=1)
        
        feature_cols = ['Car_Name', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner'] + list(poly_feature_names)
        X = df[feature_cols].values
        y = df['Selling_Price'].values.reshape(-1, 1)
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X, y, scaler, label_encoders, feature_cols, poly
    except Exception as e:
        logging.error(f"Error in data preprocessing: {str(e)}")
        raise

# Train model with k-fold cross-validation (no early stopping)
def train_model(X, y, input_size, poly):
    try:
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        best_val_r2 = -float('inf')
        best_model = None
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            logging.info(f"Training fold {fold + 1}/5")
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = CarPricePredictor(input_size)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)  # Increased lr
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25, verbose=True)
            
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train)
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)
            
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            
            num_epochs = 700  # Kept at 700
            
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                    val_predictions = val_outputs.numpy()
                    val_actual = y_val.flatten()
                    val_r2 = 1 - np.sum((val_actual - val_predictions) ** 2) / np.sum((val_actual - np.mean(val_actual)) ** 2)
                
                scheduler.step(val_loss)
                train_loss = train_loss / len(train_loader)
                logging.info(f'Fold {fold + 1}, Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val R²: {val_r2:.4f}')
                
                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    best_model = model.state_dict()
                    torch.save(best_model, 'best_model.pth')
                    logging.info(f"New best model saved with Val R²: {best_val_r2:.4f}")
        
        model.load_state_dict(torch.load('best_model.pth'))
        logging.info("Loaded best model across folds")
        return model
    except Exception as e:
        logging.error(f"Error in training: {str(e)}")
        raise

# Predict function
def predict_car_price(model, X, present_price=None):
    try:
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            raw_prediction = model(X_tensor).item()
        if present_price is not None:
            prediction = min(raw_prediction, present_price)
            if prediction < 0:
                prediction = 0.1
            return f"Predicted Selling Price: {prediction:.2f} lakhs"
        return raw_prediction
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return f"Error: {str(e)}" if present_price is not None else float('nan')

# Evaluate model
def evaluate_model(model, X_test, y_test):
    try:
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test)
            predictions = model(X_test_tensor).numpy().flatten()
            actual = y_test.flatten()
            mse = np.mean((predictions - actual) ** 2)
            rmse = np.sqrt(mse)
            r2 = 1 - np.sum((actual - predictions) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
            mae = np.mean(np.abs(actual - predictions))
            logging.info(f"Evaluation - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
            return rmse, r2, mae, predictions, actual
    except Exception as e:
        logging.error(f"Error in evaluation: {str(e)}")
        raise

# Gradio interface with metrics and visualization
def create_gradio_interface(model, label_encoders, scaler, X_test, y_test, poly):
    def predict_interface(car_name, year, present_price, kms_driven, fuel_type, seller_type, transmission, owner):
        try:
            data = {
                'Car_Name': [car_name if car_name else label_encoders['Car_Name'].classes_[0]],
                'Year': [year],
                'Present_Price': [present_price],
                'Kms_Driven': [kms_driven],
                'Fuel_Type': [fuel_type if fuel_type else label_encoders['Fuel_Type'].classes_[0]],
                'Seller_Type': [seller_type if seller_type else label_encoders['Seller_Type'].classes_[0]],
                'Transmission': [transmission if transmission else label_encoders['Transmission'].classes_[0]],
                'Owner': [owner]
            }
            df = pd.DataFrame(data)
            current_year = datetime.datetime.today().year
            df['Age'] = current_year - df['Year']
            df.drop('Year', axis=1, inplace=True)
            
            if df['Present_Price'].le(0).any():
                raise ValueError("Present_Price must be positive")
            if df['Kms_Driven'].lt(0).any():
                raise ValueError("Kms_Driven cannot be negative")
            if df['Age'].lt(0).any():
                raise ValueError("Year cannot be in the future")
            if df['Owner'].lt(0).any():
                raise ValueError("Owner cannot be negative")
            
            for col in ['Present_Price', 'Kms_Driven', 'Age']:
                Q1 = scaler.mean_[['Present_Price', 'Kms_Driven', 'Age'].index(col)] - 1.5 * scaler.scale_[['Present_Price', 'Kms_Driven', 'Age'].index(col)]
                Q3 = scaler.mean_[['Present_Price', 'Kms_Driven', 'Age'].index(col)] + 1.5 * scaler.scale_[['Present_Price', 'Kms_Driven', 'Age'].index(col)]
                df[col] = df[col].clip(Q1, Q3)
            
            categorical_cols = ['Car_Name', 'Fuel_Type', 'Seller_Type', 'Transmission']
            for col in categorical_cols:
                le = label_encoders[col]
                try:
                    df[col] = le.transform(df[col])
                except ValueError:
                    logging.warning(f"Unknown value in {col}: {df[col].iloc[0]}. Using default encoding.")
                    df[col] = le.transform([le.classes_[0]])[0]
            
            numeric_cols = ['Present_Price', 'Kms_Driven', 'Age']
            poly_features = poly.transform(df[numeric_cols])
            poly_feature_names = poly.get_feature_names_out(numeric_cols)
            df_poly = pd.DataFrame(poly_features, columns=poly_feature_names)
            df = pd.concat([df, df_poly], axis=1)
            
            feature_cols = ['Car_Name', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner'] + list(poly_feature_names)
            X = df[feature_cols].values
            X = scaler.transform(X)
            
            prediction = predict_car_price(model, X, present_price)
            return prediction, rmse_output, r2_output, mae_output, fig
            
        except Exception as e:
            logging.error(f"Gradio prediction error: {str(e)}")
            return f"Error: {str(e)}", rmse_output, r2_output, mae_output, fig

    # Evaluate model to get initial metrics and data for visualization
    rmse, r2, mae, predictions, actual = evaluate_model(model, X_test, y_test)
    rmse_output = f"RMSE: {rmse:.4f} lakhs"
    r2_output = f"R²: {r2:.4f}"
    mae_output = f"MAE: {mae:.4f} lakhs"
    
    # Create visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(actual, predictions, color='blue', alpha=0.5)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
    plt.xlabel('Actual Selling Price (lakhs)')
    plt.ylabel('Predicted Selling Price (lakhs)')
    plt.title('Actual vs Predicted Selling Prices')
    plt.grid(True)
    fig = plt.gcf()

    with gr.Blocks() as interface:
        gr.Markdown("# Car Price Prediction")
        gr.Markdown("Enter car details to predict the selling price (capped at Present Price, minimum 0.1 lakhs).")
        
        with gr.Row():
            car_name = gr.Dropdown(choices=list(label_encoders['Car_Name'].classes_), label="Car Name", value=label_encoders['Car_Name'].classes_[0])
            year = gr.Number(label="Year", value=2015, precision=0)
        
        with gr.Row():
            present_price = gr.Number(label="Present Price (lakhs)", value=5.0)
            kms_driven = gr.Number(label="Kms Driven", value=50000, precision=0)
        
        with gr.Row():
            fuel_type = gr.Dropdown(choices=list(label_encoders['Fuel_Type'].classes_), label="Fuel Type", value=label_encoders['Fuel_Type'].classes_[0])
            seller_type = gr.Dropdown(choices=list(label_encoders['Seller_Type'].classes_), label="Seller Type", value=label_encoders['Seller_Type'].classes_[0])
        
        with gr.Row():
            transmission = gr.Dropdown(choices=list(label_encoders['Transmission'].classes_), label="Transmission", value=label_encoders['Transmission'].classes_[0])
            owner = gr.Number(label="Owner (number of previous owners)", value=0, precision=0)
        
        predict_button = gr.Button("Predict Price")
        with gr.Row():
            output = gr.Textbox(label="Prediction")
            rmse_display = gr.Textbox(value=rmse_output, label="RMSE")
            r2_display = gr.Textbox(value=r2_output, label="R²")
            mae_display = gr.Textbox(value=mae_output, label="MAE")
        with gr.Row():
            viz_panel = gr.Plot(value=fig, label="Actual vs Predicted Plot")
        
        predict_button.click(
            fn=predict_interface,
            inputs=[car_name, year, present_price, kms_driven, fuel_type, seller_type, transmission, owner],
            outputs=[output, rmse_display, r2_display, mae_display, viz_panel]
        )
    
    return interface

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    X, y, scaler, label_encoders, feature_cols, poly = load_and_preprocess_data()
    
    # Split data for testing (20% holdout)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model with k-fold
    model = train_model(X_temp, y_temp, X_temp.shape[1], poly)
    
    # Evaluate model
    predictions = evaluate_model(model, X_test, y_test)
    logging.info(f"Sample Predictions (first 10):\n{list(zip(y_test[:10].flatten(), predictions[:10], [13.70, 9.40, 13.60, 0.48, 0.57, 9.90, 18.54, 22.95, 9.83, 8.61][:10]))}")
    
    # Save preprocessing objects
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('poly_features.pkl', 'wb') as f:
        pickle.dump(poly, f)
    logging.info("Saved label_encoders.pkl, scaler.pkl, and poly_features.pkl")
    
    # Launch Gradio interface
    interface = create_gradio_interface(model, label_encoders, scaler, X_test, y_test, poly)
    interface.launch()
