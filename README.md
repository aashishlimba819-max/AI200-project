# AI200 Project: Used Car Resale Price Prediction Using Linear Regression

A machine learning project that predicts the resale price of used cars based on their specifications using **Log-Transformed Linear Regression**.

## 1. Project Overview

Estimating the resale value of a used car is challenging because its price depends on several factors, including vehicle age, kilometres driven, fuel type, transmission, ownership history, engine capacity, mileage and power.

This project develops a machine learning model to predict used-car selling prices using historical vehicle data from Kaggle.

We perform exploratory data analysis, data cleaning, feature engineering and leak-free preprocessing before training and evaluating a Linear Regression model. A logarithmic transformation is applied to the target variable to handle the highly right-skewed distribution of selling prices.

## 2. Problem Statement

Used-car prices vary significantly depending on vehicle characteristics, depreciation, usage and market demand. Estimating a reasonable resale price manually can therefore be difficult.

The objective of this project is to develop a regression model that estimates the selling price of a used car using its relevant characteristics.

## 3. Project Objectives

- Analyze the factors associated with used-car resale prices.
- Perform statistical analysis and Exploratory Data Analysis (EDA).
- Identify and handle duplicate records, missing values and invalid data.
- Extract meaningful numerical features from unstructured vehicle specifications.
- Build a leak-free preprocessing pipeline.
- Implement Log-Transformed Linear Regression using Python and Scikit-learn.
- Evaluate model performance using MAE, RMSE and R².
- Visualize predictions and residuals.
- Analyze prediction errors and identify model limitations.

## 4. Dataset

**Source:** Kaggle – Vehicle Dataset from Cardekho

Dataset link: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho?resource=download

**Original dataset:** `Car details v3.csv`

### Dataset Overview

| Description | Value |
|---|---:|
| Original records | 8,128 |
| Original columns | 13 |
| Duplicate records removed | 1,202 |
| Cleaned records | 6,926 |
| Training records (80%) | 5,540 |
| Testing records (20%) | 1,386 |

### Original Dataset Features

- `name` – Car name and model
- `year` – Manufacturing year
- `selling_price` – Selling price (target variable)
- `km_driven` – Kilometres driven
- `fuel` – Fuel type
- `seller_type` – Seller category
- `transmission` – Transmission type
- `owner` – Ownership history
- `mileage` – Fuel efficiency
- `engine` – Engine capacity (CC)
- `max_power` – Maximum power (bhp)
- `torque` – Engine torque specification
- `seats` – Number of seats

## 5. Data Cleaning and Feature Engineering

The original dataset contained duplicate records, missing values, inconsistent string formats and invalid numerical entries.

The following operations were performed:

### Duplicate Removal
- Identified and removed 1,202 duplicate records.
- Reduced the dataset from 8,128 to 6,926 unique records.

### Missing Values and Invalid Entries
- Audited missing values in mileage, engine, max_power, torque and seats.
- Identified zero-value anomalies in mileage and max_power.
- Converted 15 zero-mileage records and 3 zero-power records into missing values for appropriate imputation.

### Mileage Extraction
- Extracted numerical mileage values from strings such as `23.4 kmpl`.
- Identified the mileage units `kmpl` and `km/kg`.
- Verified unit consistency across fuel categories.

### Torque Standardization
- Used regular expressions to extract numerical torque values from inconsistent strings.
- Converted torque values expressed in kgm into Newton-metres using:

  `1 kgm = 9.80665 Nm`

- Handled different torque formats containing RPM information.
- Removed physically implausible torque outliers above 700 Nm.
- Created a standardized numerical feature named `torque_nm`.

## 6. Feature Selection and Preprocessing

**Target variable:** `selling_price`

The following 11 input features were selected:

| Feature | Description |
|---|---|
| year | Manufacturing year |
| km_driven | Kilometres driven |
| fuel | Fuel type |
| seller_type | Seller category |
| transmission | Transmission type |
| owner | Ownership history |
| mileage | Fuel efficiency |
| engine | Engine capacity (CC) |
| max_power | Maximum power (bhp) |
| seats | Number of seats |
| torque_nm | Standardized torque (Nm) |

The following columns were excluded:

- `name` – Excluded to avoid high-cardinality categorical encoding.
- `torque` – Original unstructured torque column replaced by `torque_nm`.
- `mileage_unit` – Excluded after extracting the numerical mileage feature.

### Train-Test Split

- Training data: 80% (5,540 records)
- Testing data: 20% (1,386 records)
- Random state: 42

The data was split before fitting the imputation and encoding transformations to prevent data leakage.

### Preprocessing Pipeline

**Numerical features:**
- Missing values handled using median imputation.

**Categorical features:**
- Missing values handled using most-frequent imputation.
- One-hot encoding applied using `OneHotEncoder(handle_unknown='ignore')`.

The 11 selected input columns were expanded into **21 encoded model features**.

The preprocessing pipeline was fitted only on training data and then applied to the test data.

## 7. Methodology

The project follows this workflow:

1. Download the dataset from Kaggle.
2. Load and inspect the dataset.
3. Perform exploratory data analysis.
4. Remove duplicate records.
5. Identify missing values and invalid entries.
6. Extract and standardize mileage and torque features.
7. Select relevant input features and target variable.
8. Split the data into training and testing sets.
9. Build numerical and categorical preprocessing pipelines.
10. Apply a logarithmic transformation to the target variable.
11. Train the Linear Regression model.
12. Generate predictions on unseen test data.
13. Evaluate the model using regression metrics.
14. Perform residual and prediction-error analysis.

## 8. Model Architecture

### Log-Transformed Linear Regression

The project uses Scikit-learn's `LinearRegression` estimator wrapped inside `TransformedTargetRegressor`.

Since selling prices are highly right-skewed, the target variable is transformed using the natural logarithm before model training.

The model learns:

`y' = ln(y)`

where `y` represents the original selling price.

Predictions are converted back into the original price scale using:

`Predicted Price = exp(Predicted Log Price)`

The complete model combines:
- Numerical and categorical preprocessing.
- Missing-value imputation.
- One-hot encoding.
- Log transformation of the target variable.
- Linear Regression.
- Inverse transformation of predictions.

## 9. Model Evaluation

The model was evaluated on 1,386 unseen test records.

### Final Test Results

| Evaluation Metric | Result |
|---|---:|
| R² Score | 0.8516 |
| MAE | ₹1,06,764.98 |
| RMSE | ₹1,80,389.30 |

### Interpretation

- **R² = 0.8516:** The model explains approximately 85.16% of the variation in selling prices in the test dataset.
- **MAE:** The average absolute prediction error is approximately ₹1.07 lakh.
- **RMSE:** Larger prediction errors receive greater weight, resulting in an RMSE of approximately ₹1.80 lakh.

These results indicate that the model performs reasonably well on the test dataset, although prediction accuracy varies across vehicle price ranges.

## 10. Visualizations and Error Analysis

The project includes the following visualizations and diagnostics:

### Exploratory Data Analysis
- Distribution of used-car selling prices.
- Correlation heatmap of numerical features.
- Selling price versus kilometres driven.
- Relationships between vehicle characteristics and selling prices.

### Model Evaluation
- Actual versus predicted selling prices.
- Residual plot.
- Distribution of prediction errors.
- Analysis of the ten largest prediction errors.

### Error Analysis Findings

- Budget and mid-range vehicles, particularly those priced between ₹1 lakh and ₹15 lakh, generally show closer agreement between actual and predicted prices.
- Some of the largest prediction errors occur in luxury and premium vehicles, including Volvo, Lexus, BMW and Mercedes-Benz models.
- These errors highlight the limitations of using a linear model without explicitly incorporating brand-related price premiums.

## 11. Limitations

- The `name` feature was excluded because it contains a large number of unique car names and models.
- The model cannot explicitly learn brand-specific price premiums.
- Used-car depreciation and market demand may involve nonlinear relationships that a basic Linear Regression model cannot fully capture.
- The dataset represents historical listings and may not reflect current market prices.
- Prediction accuracy can vary significantly for luxury and premium vehicles.

## 12. Future Enhancements

- Develop a manual car-price prediction interface where users enter vehicle specifications and receive an estimated resale price.
- Allow users to provide the actual selling price and calculate the prediction error.
- Explore additional features such as car brand and model.
- Investigate nonlinear relationships and possible feature interactions.
- Improve model performance through further feature engineering and experimentation.

## 13. Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Preprocessing, model training and evaluation |
| Joblib | Model serialization |
| Jupyter Notebook | Interactive development |
| Git and GitHub | Version control and collaboration |

## 14. Project Structure

```text
used-car-resale-price-prediction/
│
├── data/
│   └── Car details v3.csv
│
├── notebooks/
│   ├── used_car_price_data_processing.ipynb
│   └── used_car_price_prediction.ipynb
│
├── results/
│   └── plots/
│
├── used_car_price_model.joblib
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 15. Team Members and Contributions

| Team Member | Role | Contribution |
|---|---|---|
| Aashish (2025icb1433) | Model Training and Testing | Train-test split, leak-free preprocessing pipeline and Log-Transformed Linear Regression model. |
| Yupendra Kumar (2025icb1452) | Data Preprocessing | Duplicate removal, missing-value analysis, mileage extraction and torque standardization. |
| Yogendra (2025icb1451) | Problem Selection and Direction | Problem definition, dataset selection and project methodology. |
| Bhavanam Harshavardhan Reddy (2025ICB1439) | Presentation and Synthesis | Presentation design and synthesis of project findings. |
| Nikhil (2025icb1355) | Evaluation and Visualization | Regression metrics, residual analysis, prediction-error analysis and plots. |

---

**AI200 Project – Used Car Resale Price Prediction Using Linear Regression**
