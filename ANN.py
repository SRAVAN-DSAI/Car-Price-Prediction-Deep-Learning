import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime

def preprocess_data(df):
    """
    Preprocesses the car data.

    Args:
        df: pandas DataFrame containing the car data.

    Returns:
        X: numpy array of features.
        y: numpy array of target variable (selling price).
    """
    year = datetime.datetime.today().year
    # Ensure 'Age' column is created
    df['Age'] = year - df['Year']
    df.drop('Year', axis=1, inplace=True)

    le = LabelEncoder()
    for col in ['Car_Name', 'Fuel_Type', 'Seller_Type', 'Transmission']:
        df[col] = le.fit_transform(df[col])

    X = df.drop('Selling_Price', axis=1)
    y = df['Selling_Price']

    return X.values, y.values

def visualize_data(df, output_dir="visualizations"):
    """
    Visualizes the distributions of key features in the car dataset and saves the figures.

    Args:
        df: pandas DataFrame containing the car data.
        output_dir: Directory to save the figures.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.histplot(df['Selling_Price'], bins=30, kde=True)
    plt.title('Distribution of Selling Price')
    plt.xlabel('Selling Price')
    plt.ylabel('Frequency')
    plt.savefig(f"{output_dir}/selling_price_distribution.png")
    plt.close()

    # Ensure 'Age' exists
    if 'Age' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Age', y='Selling_Price', data=df)
        plt.title('Selling Price vs. Age')
        plt.xlabel('Age (years)')
        plt.ylabel('Selling Price')
        plt.savefig(f"{output_dir}/selling_price_vs_age.png")
        plt.close()
    else:
        print("Error: 'Age' column is missing!")

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Fuel_Type', y='Selling_Price', data=df)
    plt.title('Selling Price vs. Fuel Type')
    plt.xlabel('Fuel Type')
    plt.ylabel('Selling Price')
    plt.savefig(f"{output_dir}/selling_price_vs_fuel_type.png")
    plt.close()

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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience=3):
    """
    Trains the neural network model with validation and early stopping.

    Args:
        model: PyTorch model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer.
        num_epochs: Number of training epochs.
        patience: Number of epochs with no improvement after which training will stop.

    Returns:
        train_losses: List of training losses for each epoch.
        val_losses: List of validation losses for each epoch.
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_epoch_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item() * inputs.size(0)
        train_epoch_loss /= len(train_loader.dataset)
        train_losses.append(train_epoch_loss)

        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_epoch_loss += loss.item() * inputs.size(0)
        val_epoch_loss /= len(val_loader.dataset)
        val_losses.append(val_epoch_loss)

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}')

    model.load_state_dict(best_model_state)

    return train_losses, val_losses

def evaluate_model(model, test_loader, y_test):
    """
    Evaluates the trained model on the test data.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for test data.
        y_test: True values for the test set.

    Returns:
        predictions: Predicted prices.
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs)
            predictions.extend(outputs.numpy().flatten())
    
    # Calculate RMSE and R² score
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R² Score: {r2:.4f}")

    return predictions,rmse,r2

def save_report(train_losses, val_losses, rmse, r2, report_file='model_report.txt'):
    """
    Saves a detailed report of the model's performance to a text file.

    Args:
        train_losses: List of training losses for each epoch.
        val_losses: List of validation losses for each epoch.
        rmse: RMSE value for the test set.
        r2: R² score for the test set.
        report_file: File path to save the report.
    """
    with open(report_file, 'w') as f:
        f.write("Car Price Prediction Model Report\n")
        f.write("="*40 + "\n")
        
        f.write(f"Test RMSE: {rmse:.4f}\n")
        f.write(f"Test R² Score: {r2:.4f}\n")
        
        f.write("\nTraining Losses:\n")
        for epoch, loss in enumerate(train_losses, 1):
            f.write(f"Epoch {epoch}: {loss:.4f}\n")
        
        f.write("\nValidation Losses:\n")
        for epoch, loss in enumerate(val_losses, 1):
            f.write(f"Epoch {epoch}: {loss:.4f}\n")

    print(f"Report saved to {report_file}")

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("Car-Price-Prediction-Deep-Learning/cardekho_data.csv")

    # Ensure 'Age' column is created before visualizations
    X, y = preprocess_data(df)

    # Visualize data distributions
    visualize_data(df)

    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Convert data to PyTorch tensors
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

    # Create DataLoaders
    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Create model, loss function, and optimizer
    input_size = X_train.shape[1]
    model = CarPricePredictor(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Adjusted learning rate

    # Train the model
    num_epochs = 20
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

    # Plot training and validation losses
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("training_validation_loss.png")
    plt.close()

    # Evaluate on test set and print accuracy
    predictions,rmse,r2 = evaluate_model(model, test_loader, y_test)

    # Save the trained model
    torch.save(model.state_dict(), 'car_price_predictor.pth')
    save_report(train_losses, val_losses, rmse, r2, 'model_report.txt')
