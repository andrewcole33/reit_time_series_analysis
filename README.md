# <center><ins>REIT Portfolio Time Series Analysis<ins/><center/>
### <center>Data Scientist: Andrew Cole<center/>


#### <ins>Background Information<ins/>
Real Estate Investment Trusts (REIT) act as a unique and valuable financial instrument for all types of investors. As the name inherently states, they provide investment opportunity into all things real estate. A share of a REIT can represent shared "ownership" of a physical property space, or shares of the mortgage (mREIT) income. They provide liquid & low cost access to historically high returns when compared with other asset classes such as traditional stocks, bonds, or treasury classes. REITs are tax exempt earnings where the operating party is required by law to pay out 90% of their income to investors via dividends.

#### <ins>Goals:<ins/>
For this project, I aimed to build an accurate Time Series model (SARIMAX) with the presence of exogenous variables to predict the future movements and direction of the individual REITs in the 8-REIT portfolio. This project was completed within a two-week time period.

#### <ins>Summary:<ins/>
The REIT historical data (endogenous) was obtained through AlphaVantage's API. The data ranges from 2000 - 2019 (5024 entries) and was resampled to monthly date-time intervals (240 entries). The endogenous data was then decomposed and differenced to account for the seasonality & trend present in the series. A Dickey-Fuller test for stationarity is applied to confirm that the series are, in fact, stationary.

Next, the exogenous data was gathered from FRED API(Federal Reserve Economic Database). 55 exogenous features were selected (this number is largely due to time constraints, ideally a significantly larger amount of exogenous features would be selected) and then a LASSO regression was performed for important feature selection **respective to each individual REIT**.

With the exogenous features now selected for each respective REIT, another SARIMAX model is derived and fit w/ predictions. 
