## Car-Price-Prediction-Deep-Learning


This repository implements a machine learning model to predict car prices based on features like car name, fuel type, seller type, transmission, and car age using a neural network built with PyTorch.

## Project Overview

### Steps:
1. **Data Exploration and Cleaning**
   - The dataset is loaded, and necessary cleaning steps are applied (e.g., handling missing values, transforming categorical variables).
2. **Data Preprocessing**
   - Transform categorical variables to numerical values using techniques like Label Encoding.
   - Create new features like car age by calculating it from the car's year of manufacture.
3. **Model Building**
   - A neural network model is created and trained to predict car prices.
4. **Model Evaluation**
   - The model is evaluated on a test set using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score.

## Requirements

To run this project, you need the following Python packages:

- numpy
- pandas
- torch
- scikit-learn
- matplotlib

You can install them using:

```
pip install -r requirements.txt
```
## Usage
Clone the repository:

```
git clone https://github.com/your-username/car-price-prediction.git
cd car-price-prediction
```
To train the model, run the following command:

```
python train.py
```
This will start the training process, and the model will be evaluated on the test data.

Files in the Repository
data/cardekho_data.csv: The dataset containing details of cars.
src/model.py: Defines the neural network architecture for price prediction.
src/preprocessing.py: Preprocessing steps for handling and transforming the data.
train.py: The main script for training the model.
requirements.txt: List of required Python libraries.
Evaluation Metrics
The model performance is evaluated using the following metrics:

Mean Absolute Error (MAE)
Mean Squared Error (MSE)
R² Score

### (Dependencies)
```
numpy
pandas
torch
scikit-learn
matplotlib
train.py (Main Script)
```
