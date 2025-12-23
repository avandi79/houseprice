# House Price Prediction Using Linear Regression

## Overview
This project focuses on the analysis and prediction of residential house prices using structured real estate data. The objective is to explore the relationship between property characteristics and market price through data preprocessing, exploratory data analysis, feature engineering, and regression modeling.

The study demonstrates a complete data-driven workflow, from raw data cleaning to model evaluation, following reproducible and transparent analytical practices.

## Dataset
The dataset contains residential housing records with the following attributes:
- Area (square meters)
- Number of rooms
- Parking availability
- Warehouse availability
- Elevator availability
- Address (location-based categorical feature)
- Price (target variable)

After data cleaning, the final dataset consists of **3,433 observations**.

## Data Preprocessing
Key preprocessing steps include:
- Cleaning and converting the `Area` column from string to numeric format
- Removing invalid or unrealistic area values
- Handling missing values in the `Address` column
- Filtering out inconsistent records
- Creating a clean analytical dataset for modeling

## Feature Engineering
- Numerical features: Area, Room count, Parking, Warehouse, Elevator
- Categorical feature encoding: Address encoded using one-hot encoding
- Final feature matrix contains 33 numerical predictors

## Exploratory Data Analysis
- Distribution analysis using histograms
- Scatter plot visualization to examine the relationship between area and price
- Identification of price dispersion across different property sizes

## Modeling Approach
A Linear Regression model was implemented using scikit-learn:
- Dataset split into training (80%) and testing (20%) subsets
- Model trained on numerical and encoded categorical features
- Performance evaluated using R² score and Root Mean Squared Error (RMSE)

## Results
- **R² Score:** ~0.69
- **RMSE:** ~3.7 × 10⁹ (local currency units)

The results indicate a strong linear relationship between property features and housing prices, while also highlighting the limitations of linear models in capturing extreme price variations.

## Tools & Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Reproducibility
To run the project:
1. Clone the repository
2. Install required libraries
3. Open `houseprice.ipynb` in Jupyter Notebook
4. Execute cells sequentially

## Author
Fatemeh Naghdi 

