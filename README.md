# stockVortex is a stock prediction app
Creating a stock price prediction app with Streamlit involves integrating machine learning models for predictions,
providing a user interface for interaction, and displaying the results. Below is a high-level description of how you can create such an app:

1. Setup and Dependencies:
   - Install Python libraries, including Streamlit, pandas, NumPy,etc.These libraries are mentioned in Requirements.txt.

2. Data Collection and Preprocessing:
   - Gather historical stock price data for the selected stock(s).
   - Preprocess the data, including cleaning, feature engineering, and formatting it for model training.

3. Machine Learning Model:
   - Choose a suitable machine learning model for stock price prediction (e.g., linear regression, ARIMA, LSTM, etc.).In this project, SARIMAX model is used.
   - Train the model using historical stock data.

4. Streamlit Application:
   - Create a Python script (e.g., `app.py`) to set up the Streamlit application.
   - Import necessary libraries and the trained machine learning model.

5. User Interface:
   - Design the Streamlit UI to allow users to input parameters (e.g., stock symbol, date range, etc.).
   - Create input elements like text boxes, date pickers, and buttons.
  
6. Prediction Logic:
   - Implement the prediction logic using the trained machine learning model.
   - Use the user's inputs as features for prediction.

7. Display Predictions:
   - Showcase the predicted stock prices using Streamlit components like graphs (e.g., line chart) to visualize the predictions.

8. Run the App:
   - Run the Streamlit app using the `streamlit run app.py` command.


