# AI200-project
# Used Car Resale Price Using linear regression

# 1. Project overview
This project is to develop a machine learning model that predicts the resale price of a used car based on its characteristics.
The model will analyze historical vehicle data to understand how different factors influence a car's market value.

The resale price of a used car depends on serval factors, including the age of the vechile, kilometres driven, fuel type, transmission type, ownership history and present, market value.
The project aims to develop a Linear Regression machine learning model that predicts the selling price of a used car based on relevant vehicle characteristics.
Using historical vehicle data from kaggle, we will analyze relationships between car features and selling prices, train a regression model, and evalute its performance on unseen data.

# 2. Problem Statement
Estimating the resale value of a used car can be different because different vehicles have different characteristics and market values.

# 3.Project Objectives
Analyze the factors associated with used-car resale prices.

Perform statistical analysis and exploratory data analysis.

Clean and preprocess the dataset before model training.

Implement Linear Regression using Python and scikit-learn.

Evaluate model performance using appropriate regression metrics.

Visualize actual and predicted prices and interpret the results.

# 4.Dataset
Source: Kaggle

https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho?resource=download

The dataset contains used car information, including vehicle characteristics and selling prices.

The exact features used in the model will depend on the columns available in the downloaded dataset.

# Factore to Analyze

# 5. Methodology

The project will follow these steps:

Data collection: Download the dataset from Kaggle.

Data exploration: Inspect the dataset structure, feature types, summary statistics, and distributions.

Data preprocessing: Check for missing values, duplicate records, outliers, and categorical variables.

Feature selection: Select appropriate input features and define the selling price as the target variable.

Data splitting: Divide the dataset into training and testing sets.

Model training: Train a Linear Regression model on the training data.

Prediction: Generate selling-price predictions for the test data.

Evaluation: Calculate regression metrics and create relevant plots.

Interpretation: Explain the results, prediction errors, and limitations.

# 6. Technologies and Libraries

Python

Pandas — data manipulation

NumPy — numerical computation

Matplotlib — data visualization

Seaborn — statistical visualization

Scikit-learn — model training and evaluation

Jupyter Notebook — interactive development

Git and GitHub — version control and collaboration

# 7. Model Evaluation

The model will be evaluated using the following metrics:

Mean Absolute Error (MAE): Measures the average absolute difference between actual and predicted prices.

Root Mean Squared Error (RMSE): Measures prediction error while penalizing larger errors more strongly.

Coefficient of Determination (R²): Measures how much of the variation in the target is explained by the model.

Training and testing performance will be compared to assess generalization.

# 8. Expected Visualizations

The project may include:

Distribution of used-car selling prices.

Correlation heatmap of numerical features.

Selling price versus vehicle age.

Selling price versus kilometres driven.

Actual versus predicted selling prices.

Residual plot showing prediction errors.

# 9. Project Structure

used-car-resale-price-prediction/
│
├── data/
│   └── car_data.csv
│
├── notebooks/
│   └── used_car_price_prediction.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── evaluation.py
│
├── results/
│   └── plots/
│
├── requirements.txt
├── README.md
└── .gitignore


Team members
1. Aashish  2025icb1433
2. yupendra 2025icb1452
3. yogendra 2025icb1451
4. BHAVANAM HARSHAVARDHAN REDDY 2025ICB1439
5. Nikhil 2025icb1355
