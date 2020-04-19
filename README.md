# <center><ins>REIT Portfolio Time Series Analysis<ins/><center/>
### <center>Data Scientist: Andrew Cole<center/>





- See FR_TS.ipynb for detailed notebook with commentary

- Please click for a blog post which walks through the entire project: https://medium.com/p/time-series-analysis-of-a-reit-portfolio-6ec082f9fe3c?source=email-cb2c8eb9bc8--writer.postDistributed&sk=3475b07f3f1e06913c1197dedb66c455




#### <ins>Background Information<ins/>
Real Estate Investment Trusts (REIT) act as a unique and valuable financial instrument for all types of investors. As the name inherently states, they provide investment opportunity into all things real estate. There are two types of REITs: Equity & Mortgage (mREIT). An equity REIT can represent shared "ownership" of a physical property space, while shares of an mREIT equate to owning a share of the *financing* of a mortgage. They provide liquid & low cost access to historically high returns when compared with other asset classes such as traditional stocks, bonds, or treasury classes. REITs are tax exempt earnings where the operating party is required by law to pay out 90% of their income to investors via dividends.

#### <ins>Goals:<ins/>
For this project, I aimed to build an accurate Time Series model (SARIMAX) with the presence of exogenous variables to predict the future movements and direction of the individual REITs in the 8-REIT portfolio. Current trading practice in industry can be conducted using more complex deep learning algorithms (such as GRU-RNNs or LSTM Models), however this project was completed within 1.5 week deadline so the SARIMAX model will suffice.

#### <ins>Summary:<ins/>
The REIT historical data of 8 different REITs (endogenous) were obtained through AlphaVantage's API. The data ranges from 2000 - 2019 (5024 entries) and was resampled to monthly date-time intervals (240 entries). The endogenous data was then decomposed with a natural log, and then first order differenced to account for any seasonality, trend, or residual distribution variance which may be present in the respective series. A Dickey-Fuller test for stationarity is applied to confirm that the series are, in fact, stationary. After the stationarity is confirmed, the autocorrelation functions and partial-autocorrelation functions were plotted and analyzed for the selection of the moving average and autoregressive orders, respectively, to be included in the SARIMAX model parameters.

Next, the exogenous data was gathered from FRED API(Federal Reserve Economic Database). Exogenous variables were selected from six different Money & Banking categories. 55 features were selected (this number is largely due to time constraints of the project, ideally a significantly larger amount of exogenous features would be included) and then a LASSO regression was performed for important feature selection **respective to each individual REIT**.

With the exogenous features now selected for each respective REIT, another SARIMAX model is executed and fit with predictions using the model parameters already obtained previously and the new exogenous variables included.

In an attempt to scale up the robustness of the model and increase model accuracy, a GRU RNN (Gated Recurrent Unit Recursive Neural Network) model was also built. This model requires significantly more computational power to execute more powerful calculations. For this reason, this notebook was hosted through Google Cloud's web services so that I could have access to the computational power of it's GPUs. This model will be designed to learn which statistical events in the data history are important, and unlike the SARIMAX models, it can continuously re-compute according to  what it deems to be the most import events while "forgetting" what is unecessary. This model performs well in the training set however it severely suffers from a lack of data necessary for full efficiency of the model. For this reason, this GRU model will remain purely as a practice model and is not to be implemented in the business world.

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
    
    - GRU_model.ipynb


- EDA.ipynb
  - Data gathering and exploration of exogenous variables


- module.py
  - Python module containing functions for cleaning, manipulating, and execution of SARIMAX models
