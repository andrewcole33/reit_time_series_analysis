# <center><ins>REIT Portfolio Time Series Analysis<ins/><center/>
### <center>Data Scientist: Andrew Cole<center/>





- See FR_TS.ipynb for detailed notebook walkthrough





#### <ins>Background Information<ins/>
Real Estate Investment Trusts (REIT) act as a unique and valuable financial instrument for all types of investors. As the name inherently states, they provide investment opportunity into all things real estate. There are two types of REITs: Equity & Mortgage (mREIT). An equity REIT can represent shared "ownership" of a physical property space, while shares of an mREIT equate to owning a share of the *financing* of a mortgage. They provide liquid & low cost access to historically high returns when compared with other asset classes such as traditional stocks, bonds, or treasury classes. REITs are tax exempt earnings where the operating party is required by law to pay out 90% of their income to investors via dividends.

#### <ins>Goals:<ins/>
For this project, I aimed to build an accurate Time Series model (SARIMAX) with the presence of exogenous variables to predict the future movements and direction of the individual REITs in the 8-REIT portfolio. Current trading practice in industry can be conducted using more complex deep learning algorithms (such as GRU-RNNs or LSTM Models), however this project was completed within 1.5 week deadline so the SARIMAX model will suffice.

#### <ins>Summary:<ins/>
The REIT historical data of 8 different REITs (endogenous) were obtained through AlphaVantage's API. The data ranges from 2000 - 2019 (5024 entries) and was resampled to monthly date-time intervals (240 entries). The endogenous data was then decomposed with a natural log, and then first order differenced to account for any seasonality, trend, or residual distribution variance which may be present in the respective series. A Dickey-Fuller test for stationarity is applied to confirm that the series are, in fact, stationary. After the stationarity is confirmed, the autocorrelation functions and partial-autocorrelation functions were plotted and analyzed for the selection of the moving average and autoregressive orders, respectively, to be included in the SARIMAX model parameters.

Next, the exogenous data was gathered from FRED API(Federal Reserve Economic Database). Exogenous variables were selected from six different Money & Banking categories. 55 features were selected (this number is largely due to time constraints of the project, ideally a significantly larger amount of exogenous features would be included) and then a LASSO regression was performed for important feature selection **respective to each individual REIT**.

With the exogenous features now selected for each respective REIT, another SARIMAX model is executed and fit with predictions using the model parameters already obtained previously and the new exogenous variables included.

### <ins>Included in this Repository:<ins/>
- 8 REIT Time Series Models:
    - AMT_TS.ipynb
    - ELS_TS.ipynb
    - PLD_TS.ipynb
    - FR_TS.ipynb (**Contains Technical Summary**)
    - MAA_TS.ipynb
    - SUI_TS.ipynb
    - BXMT_TS.ipynb
    - RHP_TS.ipynb


- EDA.ipynb
  - Data gathering and exploration of exogenous variables


- module.py
  - Python module containing functions for cleaning, manipulating, and execution of SARIMAX models
