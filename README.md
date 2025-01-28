# Car Price Prediction with Neural Network

This project uses a neural network to predict the selling price of used cars based on various features such as age, fuel type, seller type, and more.

**Objective:** Build a machine learning model to accurately predict the selling price of used cars.

## 1. Methodology:
    * Data preprocessing: Data cleaning, feature engineering (creating 'Age' feature), and encoding categorical variables.
    * Exploratory Data Analysis (EDA): Visualizing data distributions and relationships between features.
    * Model development: Training a neural network with multiple layers.
    * Model evaluation: Evaluating model performance using metrics like RMSE and R2. 
    * Model saving: Saving the trained model for future use.

## 2. Data

* **Source:** `cardekho_data.csv` (assumed to be in the same directory)
* **Features:**
    * `Car_Name`
    * `Year`
    * `Selling_Price`
    * `Present_Price`
    * `Kms_Driven`
    * `Fuel_Type`
    * `Seller_Type`
    * `Transmission`
    * `Owner` 

## 3. Installation 

1. **Install required libraries:**
   ```bash
   pip install torch torchvision scikit-learn pandas numpy matplotlib seaborn
## 4. Usage

1. **Run the script:**
   ```bash
   python car_price_prediction.py
## 5. Code Structure

* **`preprocess_data(df)`:**
    * Creates 'Age' feature.
    * Encodes categorical features (`Car_Name`, `Fuel_Type`, `Seller_Type`, `Transmission`) using `LabelEncoder`.
    * Splits data into features (X) and target variable (y).
* **`visualize_data(df)`:**
    * Creates visualizations:
        * Distribution of `Selling_Price`.
        * Relationship between `Selling_Price` and `Age`.
        * Relationship between `Selling_Price` and `Fuel_Type`.
    * Saves visualizations to the `visualizations` directory.
* **`CarPricePredictor(nn.Module)`:**
    * Defines the neural network architecture:
        * 3 fully connected layers with ReLU activation.
* **`train_model(...)`:**
    * Trains the model with early stopping for improved performance.
    * Calculates and stores training and validation losses.
* **`evaluate_model(...)`:**
    * Evaluates the trained model on the test set.
    * Calculates and prints RMSE and R² score.
* **`save_report(...)`:**
    * Saves a detailed report of model performance to `model_report.txt`.

## 6. Model Training and Evaluation

* **Training:**
    * Uses Adam optimizer with an adjusted learning rate.
    * Trains for a specified number of epochs with early stopping.
    * Plots training and validation losses for visualization.
* **Evaluation:**
    * Calculates and prints RMSE and R² score on the test set.

## 7. Model Saving

* Saves the trained model parameters to `car_price_predictor.pth` for future use.

## 8. Results

* The final RMSE and R² score are printed in the console and saved in `model_report.txt`.
* The trained model is saved in `car_price_predictor.pth`.
