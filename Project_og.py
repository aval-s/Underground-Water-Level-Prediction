from tkinter import *
import tkinter as tk

import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

home = Tk()

# Download csv file from resources and put it in working directory
df = pd.read_csv('Water_data_Treated_og.csv', parse_dates=['QUARTER'], index_col='QUARTER')
df.head()

def graph():
    #taking dataset of ludhiana district only
    dist_df = pd.DataFrame(df['LEVEL'][df.DISTRICT == 'Ludhiana'])
    
    
    aa = auto_arima(dist_df,
                    start_p=0, start_q=0,d=0, max_p=6, max_q=6, max_d=2,
                    start_P=0, start_Q=0,D=0, max_P=6, max_Q=6, max_D=2, m=3,
                    error_action='ignore', suppress_warnings=True) 
    
    
    model = SARIMAX(dist_df, order=aa.order, seasonal_order=aa.seasonal_order)
    model_fit = model.fit()
    
    
    pred = model_fit.get_prediction(start=pd.to_datetime('2016-01-01'),end =pd.to_datetime('2020-01-01'), dynamic=False)
    pred_ci = pred.conf_int()
    ax = dist_df['1990':'2015'].plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='5 year Forecast', alpha=.8, figsize=(14, 7))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='pink', alpha=.8)
    ax.set_xlabel('Date')
    ax.set_ylabel('Furniture Sales')
    plt.legend()
    plt.show()

Label(home, text = 'Select State    :').grid(row=1, column=0)
Label(home, text = 'Select District :').grid(row=2, column=0)
b = Button(home, text="Run", command=graph)
b.grid(row=3, column=0)




home.mainloop()


