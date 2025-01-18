# remove this comment oo this is the full blown solution
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def prepare_data(df):
    """
    Prepare the data for modeling by handling missing values and encoding categorical variables
    """
    
    df_clean = df.copy()
    
    
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_columns] = df_clean[numeric_columns].fillna(df_clean[numeric_columns].mean())
    
    
    categorical_columns = df_clean.select_dtypes(include=['object']).columns
    df_clean = pd.get_dummies(df_clean, columns=categorical_columns)
    
    return df_clean

def train_sales_model(df, target_column='TotalSalesValue'):
    """
    Train a linear regression model to predict sales
    
    Parameters:
    df (pandas.DataFrame): Input dataset
    target_column (str): Name of the target variable column
    
    Returns:
    tuple: Trained model, feature scaler, predictions, and evaluation metrics
    """
    
    df_clean = prepare_data(df)
    
    
    if 'ReportID' in df_clean.columns:
        df_clean = df_clean.drop('ReportID', axis=1)
    
    
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    
    y_pred = model.predict(X_test_scaled)
    
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    return model, scaler, y_test, y_pred, rmse, r2, feature_importance

def plot_predictions(y_true, y_pred):
    """
    Create a scatter plot of predicted vs actual values
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Sales Value')
    plt.ylabel('Predicted Sales Value')
    plt.title('Actual vs Predicted Sales Values')
    plt.tight_layout()
    plt.show()


try:
    
    df = pd.read_csv('For_Prediction.csv')
    
    
    model, scaler, y_test, y_pred, rmse, r2, feature_importance = train_sales_model(df)
    
    
    print("\nModel Performance Metrics:")
    print(f"Root Mean Square Error: {rmse:.2f}")
    print(f"R-squared Score: {r2:.4f}")
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    
    plot_predictions(y_test, y_pred)

except FileNotFoundError:
    print("Error: 'For_Prediction.csv' not found. Please ensure the data file exists.")
except Exception as e:
    print(f"An error occurred: {str(e)}")