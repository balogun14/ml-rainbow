import pandas as pd
import numpy as np


df = pd.read_csv('For_Prediction.csv')

def simple_linear_regression(x, y):
    """Calculate slope and intercept for linear regression"""
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    slope = numerator / denominator
    intercept = y_mean - (slope * x_mean)
    
    
    predictions = (slope * x) + intercept
    
    
    ss_tot = np.sum((y - y_mean) ** 2)
    ss_res = np.sum((y - predictions) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return slope, intercept, r_squared




print("Predicting TotalSalesValue using Quantity features:\n")

df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
x = df['Quantity'].values
y = df['TotalSalesValue'].values

slope, intercept, r_squared = simple_linear_regression(x, y)

print(f"\nUsing {'Quantity'} as predictor:")
print(f"Equation: TotalSalesValue = {intercept:.2f} + {slope:.2f} × {'Quantity'}")
print(f"R² Score: {r_squared:.4f}")


example_value = np.median(x)  
predicted_sale = (slope * example_value) + intercept
print(f"Example: For {'Quantity'} = {example_value:.0f}, Predicted TotalSalesValue: {predicted_sale:.2f}")