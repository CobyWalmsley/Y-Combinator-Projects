#So I want a function that takes 2 vectors 
#Read my pandas file
#pull the two dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import ttest_rel, ttest_ind
df=pd.read_excel("C:/Users/cobyw/OneDrive/Documents/Voyage Analytics/MFE/Cross_Correlation.xlsx")
inflation=df["inflation-condition"]
rate=df["Inflation Rate"]


def newey_graph():
    months=12
    corrs=[]
    lags = list(range(503)) #1 year
    lags.remove(0)
    for lag in lags: 
        if lag !=0:
            new_inflation=inflation[0:-lag] #this setup cuts the end off of the sentiment and shifts the rate forward. Correct.
            new_rate=rate[lag:]
            corr = np.corrcoef(np.array(new_inflation),np.array(new_rate))[0,1]
            corrs.append(corr)
    
    plt.title("Correlation vs Lag")
    plt.plot(np.array(lags),np.array(corrs))
    yticks_positions = [21, 42, 63, 84, 105, 126,147,168,189,210,231,251,502]
    yticks_labels = ['1','2', '3','4','5','6','7', '8', '9', '10', '11', '12','24']
    plt.xlabel("Months Lag")
    plt.ylabel("Correlation")
    plt.xticks(yticks_positions, yticks_labels)
    plt.show()
    

def t_score():
    model = model.get_robustcov_results(cov_type="HAC", maxlags=5)
    t_stat_paired, p_value_paired = ttest_rel(inflation, rate)
    t_stat_ind, p_value_ind = ttest_ind(inflation, rate)
    print(t_stat_ind)

newey_graph()