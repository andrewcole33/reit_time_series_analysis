import pandas_datareader.data as web
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from numpy import hstack
import warnings
import os
from fredapi import Fred
warnings.filterwarnings('ignore')

import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


def decomposition(df):
"""
This function will perform decomposition of a passed time series indexed dataframe. It will take a log of the time series data, then decompose to account for trend, seasonality, and residual distribituion.
"""
        ts = df
        decomposition = seasonal_decompose(np.log(ts))
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        # Plot with subplots
        plt.figure(figsize=(10,6))
        plt.subplot(411)
        plt.plot(np.log(ts), label='Original', color="blue")
        plt.legend(loc='best')
        plt.subplot(412)
        plt.plot(trend, label='Trend', color="blue")
        plt.legend(loc='best')
        plt.subplot(413)
        plt.plot(seasonal,label='Seasonality', color="blue")
        plt.legend(loc='best')
        plt.subplot(414)
        plt.plot(residual, label='Residuals', color="blue")
        plt.legend(loc='best')
        plt.tight_layout()
        
        
        

def dickey_fuller(df):
"""
Performs a dickey-fuller statistical test to verify that a time series is stationary and all trend has been removed. 

Input: DateTime indexed DataFrame

Output: A P-value for Dickey-Fuller hypothesis test.
"""
    dftest = adfuller(df['mid'])
    dfoutput = pd.Series(dftest[0:4], index = ['Test Stat', 'p-value', '# lags used', '# Observations Used'])
    dftest_pvalue = dftest[1]
    
    if dftest_pvalue < 0.05:
        print(f"The series passes the Dickey Fuller Test for Stationarity. P-Value = {dftest_pvalue}")
    else:
        print(f"The series fails the Dickey Fuller Test. P-Value = {dftest_pvalue}")
        
        
        
        
def arima_endog(df,observed,parameters, seasonal_parameters):
    arima_model = sm.tsa.statespace.SARIMAX(df,
                                            order = parameters,
                                            seasonal_order = seasonal_parameters,
                                            enforce_stationarity = False,
                                            enforce_invertability = False,
                                            trend = 't')
    results = arima_model.fit()
    print(f'Summary: {results.summary()}')
    print()
    print('Diagnostics: ')
    results.plot_diagnostics(figsize = (15,8))
    plt.show()
    
    print('--------------------------------------------------------------------------------------------------------------------------------------------------------------')
    print(' ')
    print('Predictions vs. Observed: ')
    print(' ')
    
    predictions = results.get_prediction(start = pd.to_datetime('2016-01-01'), end = pd.to_datetime('2020-01-01'), dynamic = False)
    pred_conf = predictions.conf_int()
    
    #Plot observed values
    font = {'family':'sans-serif',
        'weight':'bold',
        'size':12}
    matplotlib.rc('font', **font)
    ax = observed['2000-01-01':].plot(label = 'observed', figsize = (20,5))
    
    #Plot predicted values
    predictions.predicted_mean.plot(ax = ax, label = 'One Step Ahead Forecast', alpha = .9, color = 'darkblue', style = '--')
    
    #Plot range of confidence interval
    ax.fill_between(pred_conf.index,
                    pred_conf.iloc[:,0],
                    pred_conf.iloc[:,1],
                    color = 'lightblue', alpha =.7)
    
    #Set Axes Labels
    ax.set_xlabel('Year')
    ax.set_ylabel('Mid Price')
    plt.legend(loc = 'upper left')
    plt.title('Mid Price Over Time (Observed & Model Predicted)')
    plt.tight_layout()
    plt.show()

    
def arima_exog(df_train, df, parameters, seasonal_parameters, exog_train, exog_test):
    arima_model = sm.tsa.statespace.SARIMAX(df_train,
                                            exog = exog_train,
                                            order = parameters,
                                            seasonal_order = seasonal_parameters,
                                            enforce_stationarity = False,
                                            enforce_invertability = False,
                                            trend = 't')
    results = arima_model.fit()
    print(f'Summary: {results.summary()}')
    print()
    print('Diagnostics: ')
    results.plot_diagnostics(figsize = (15,8))
    plt.show()
    
    print('--------------------------------------------------------------------------------------------------------------------------------------------------------------')
    print(' ')
    print('Predictions vs. Observed: ')
    print(' ')
    
    predictions = results.get_prediction(
                                         start = pd.to_datetime('2015-01-01'), 
                                         end = pd.to_datetime('2020-01-01'), 
                                         dynamic = False, 
                                         exog = exog_test)
    pred_conf = predictions.conf_int()
    
    #Plot observed values
    ax = df['2000-01-01':].plot(label = 'observed', figsize = (17,5))
    
    #Plot predicted values
    predictions.predicted_mean.plot(ax = ax, label = 'One Step Ahead Forecast', alpha = .9, color = 'darkblue', style = '--')
    
    #Plot range of confidence interval
    ax.fill_between(pred_conf.index,
                    pred_conf.iloc[:,0],
                    pred_conf.iloc[:,1],
                    color = 'lightblue', alpha =.7)
    
    #Set Axes Labels
    ax.set_xlabel('Year')
    ax.set_ylabel('Mid Price')
    plt.legend(loc = 'upper left')
    plt.title('Mid Price Over Time (Observed & Model Predicted)')
    plt.tight_layout()
    plt.show()
    

    
def rmse_cv(model, X, y):
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)



def lasso_reg(dataframe):
    
    target = dataframe['mid']
    dataframe = dataframe.drop(['mid'], axis = 1)
    
    scale = MinMaxScaler()
    rates_trans = scale.fit_transform(dataframe)
    
    model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(dataframe, target)
    rmse_cv(model_lasso, dataframe, target).mean()
    
    print(f"The R-score of the Lasso Regression is: {model_lasso.score(dataframe, target)}")
    print('')
    
    coef = pd.Series(model_lasso.coef_, index = dataframe.columns)
    
    imp_coef = pd.concat([coef.sort_values().head(26),
                          coef.sort_values().tail(26)])
    matplotlib.rcParams['figure.figsize'] = (8,8)
    imp_coef.plot(kind = 'barh')
    plt.title("Coefficients in the Lasso Model")
    plt.show()
    
    matplotlib.rcParams['figure.figsize'] = (8,8)
    preds = pd.DataFrame({"preds":model_lasso.predict(dataframe), "true":target})
    preds["residuals"] = preds["true"] - preds["preds"]
    preds.plot(x = "preds", y = "residuals",kind = "scatter")
    
    
    
    
