from tkinter import *
import tkinter as tk
from tkinter import ttk

import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

import warnings
warnings.filterwarnings('ignore')

plt.style.use(['ggplot','dark_background'])

home = Tk()

df = pd.read_csv('Water_data_Treated_og.csv', parse_dates=['QUARTER'], index_col='QUARTER')

states = list(df.STATE.unique())
print(states)
dist_sel=""
state_sel=""
dist_df=""


def options1(event):
    global dist_combo
    global state_sel
    state_sel = state_combo.get()
    dists = list((df.DISTRICT[df.STATE==state_sel].unique()))
    print(state_sel)
    print(dists)
    print(type(dists))
    dist_combo = ttk.Combobox(home, value=dists)
    dist_combo.bind("<<ComboboxSelected>>", options2)
    dist_combo.grid(row=2,column=1)

def options2(event):
    global dist_sel
    global dist_combo
    dist_sel = dist_combo.get()
    print("inside options", dist_sel)

    
def graph():
    global dist_df
    print("Inside graph",dist_sel)
    dist_df = pd.DataFrame(df['LEVEL'][df.DISTRICT == dist_sel])
    
    
    aa = auto_arima(dist_df,
                    start_p=0, start_q=0,d=0, max_p=6, max_q=6, max_d=2,
                    start_P=0, start_Q=0,D=0, max_P=6, max_Q=6, max_D=2, m=3,
                    error_action='ignore', suppress_warnings=True) 
    
    
    model = SARIMAX(dist_df, order=aa.order, seasonal_order=aa.seasonal_order)
    model_fit = model.fit()
    
    
    pred = model_fit.get_prediction(start=pd.to_datetime('2016-01-01'),
                                    end =pd.to_datetime('2020-01-01'),
                                    dynamic=False)
    pred_ci = pred.conf_int()
    ax = dist_df['1990':'2015'].plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='5 year Forecast', alpha=.8, figsize=(10, 5))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='pink', alpha=.8)
    ax.set_xlabel('Date')
    ax.set_ylabel('Underground Water Level')
    plt.legend(loc='upper left', fancybox=True)
    tit = str("State: " + state_sel + ";  District: " + dist_sel)
    plt.title(tit)
    plt.show()
    del(aa)
    del(model)
    del(model_fit)


home.title('Underground Water Level Predition')
Label(home, text = 'Select State    :').grid(row=1, column=0)
state_combo = ttk.Combobox(home, value=states)
state_combo.bind("<<ComboboxSelected>>", options1)
state_combo.grid(row=1,column=1)

Label(home, text = 'Select District :').grid(row=2, column=0)
dist_combo = ttk.Combobox(home, value=['Select state first'])
dist_combo.grid(row=2,column=1)

tk.Label(home,text='').grid(row=3)

b = Button(home, text="Run", command=graph, width=15)
b.grid(row=4, column=0)

e = Button(home, text="Exit", command=home.destroy, width=15)
e.grid(row=4, column=1)

tk.Label(home,text='').grid(row=5)

tb = tk.Text(home,width=60,height=16, bg='black', fg='green', spacing1=5)
tb.grid(row=6,columnspan=2)

tb.config(state='normal')
tb.delete('1.0','end')
tb.insert('end', "#### Underground water level predictor ####")
tb.insert('end', "\n\nInstructions: \nStep 1 : Select state then select district")
tb.insert('end', "\nStep 2 : Click on run and get a 5 year forecast")
tb.insert('end', "\n\nNotes : \nSARIMA model is being trained on data from 1990 to 2015, \nprediction is done for years 2016 to 2020.")
tb.insert('end', "\n\nTechnical Project for CDAC\nMade By : \nAvalvir Sekhon\nAnjali Jaiswani\nAiman Ara\nGitanshi Bhutani")
tb.config(state='disabled')

home.mainloop()


