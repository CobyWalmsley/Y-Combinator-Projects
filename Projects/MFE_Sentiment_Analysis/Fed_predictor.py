import numpy as np
import pandas as pd
import scipy.stats as stats

df = pd.read_excel('C:/Users/cobyw/OneDrive/Documents/Voyage Analytics/MFE/Fed_prediction.xlsx')

x1=df['X1']
x2=df['X2']
x3=df['X3']
y=df['Y']


def create_model(x1,y):
    mean_x1=np.mean(x1)
    mean_y=np.mean(y)

    beta1 = np.sum((x1 - mean_x1) * (y - mean_y)) / np.sum((x1 - mean_x1) ** 2)
    beta0 = mean_y - beta1 * mean_x1
    prediction = []
    for x in x1:
        y_predicted=beta0+beta1*x
        prediction.append(y_predicted)
    y_prediction=np.array(prediction)

    n = len(x1)
    residuals = y - y_prediction
    squared_errors = residuals ** 2
    sigma_squared = np.sum(squared_errors) / (n - 2)  # Variance of residuals
    se_beta1 = np.sqrt(sigma_squared / np.sum((x - mean_x1) ** 2))  # Standard error of beta1
    t_value = stats.t.ppf(0.975, df=n-2)  # Two-tailed t-value for 95% confidence   
    conf_interval = (beta1 - t_value * se_beta1, beta1 + t_value * se_beta1)
    print("beta0: ",beta0)
    print("beta1: ",beta1)
    print("T value: ",t_value)
    print("95% Confidence Interval: ",conf_interval)
    print("Squared Error: ", squared_errors)
    return prediction,beta0,beta1,squared_errors,sigma_squared,se_beta1,t_value,conf_interval



frame=create_model(x2,y)[0]

df2=pd.DataFrame(frame)
df2.to_excel("Predictions 2.xlsx")