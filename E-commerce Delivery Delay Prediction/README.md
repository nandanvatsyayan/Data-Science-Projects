# E-commerce-Delivery-Delay-prediction-Using-Machine-learning

# **Introduction**

This project predicts the number of days a product delivery might be delayed using real-world e-commerce data from the Olist dataset. I used a Stacking Regressor to combine multiple models for better performance.

# **Requirements**

- Pandas
- Numpy
- Scikit-learn
- Xgboost
- Matplotlib
- Seaborn
- Joblib
- Streamlit

# **Installations**

- pip install pandas
- pip install numpy
- pip install scikit-learn
- pip install xgboost
- pip install matplotlib
- pip install seaborn
- pip install joblib
- pip install streamlit
- pip install pickle


# **Dataset**

The dataset used is a merged E-commerce dataset stored in the file:
combined_data.csv

It includes customer, order, product, payment, and delivery-related features such as:
- Order timestamps
- Freight value
- Product weight and volume
- Payment details
- Delivery timelines

# **Exploratory Data Analysis**

I explored the Olist E-commerce dataset to understand patterns, trends, and relationships in the data. Here are the key steps and visualizations used:

- Missing Value Heatmap: Checked for missing values using heatmap.
- Target Distribution: Plotted histogram and boxplot for delivery_delay.
- Correlation Heatmap: Identified relationships between numerical features.
- Top-Selling Categories: Barplot of most sold product categories.
- Top States by Orders: Barplot showing states with the most orders.
- Delivery Time Analysis: Compared actual vs estimated delivery times.
- These visualizations helped in selecting relevant features and understanding delivery delays.

# **Model Used**
I trained a Stacking Regressor with the following components:

*Base Models:*
- XGBRegressor (with best parameters from RandomizedSearchCV)
- GradientBoostingRegressor

*Meta Model:*
- LinearRegression

# **Model Performance**
Evaluated on the test data using these metrics:

*Metric	Value*
- MAE (Mean Absolute Error): 4.33 days
- RMSE (Root Mean Squared Error): 7.44 days
- R² Score: 0.4609
The target variable **(delivery_delay)** shows a normal distribution, indicating the model isn't biased and is generalizing fairly well.

# **Streamlit App**
I built a simple interactive Streamlit app where users can input key order details and predict delivery delay in real-time.

- App file: Main.py
- Model file: final_stacked_model.pkl

*Example output from the app:* 🚚 Predicted Delivery Delay: -6.89 days.

Negative delay implies that the product may arrive earlier than expected.

