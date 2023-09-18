import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date,timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

#title 

st.markdown(""" <style> .font {
font-size:50px ; font-weight: bold; font-family: 'Courier New'; color: #DB7093;} 
</style> """, unsafe_allow_html=True)
st.markdown('<p class="font">StockVortex</p>', unsafe_allow_html=True)

st.write("<p style='color:LightPink  ; font-size: 20px;font-family: 'Garamond' ;font-weight: normal;'>Where stocks converge and profits swirl – welcome to StockVortex!</p>",unsafe_allow_html=True)
#st.subheader('Where stocks converge and profits swirl – welcome to StockVortex!')
st.image("https://www.prococommodities.com/wp-content/uploads/2021/02/blog-03_1024x768_acf_cropped.jpg")

#take user input
st.sidebar.header('Parameters')
start_date=st.sidebar.date_input('Start Date',date(2020,1,1))
end_date=st.sidebar.date_input('End Date',date(2020,12,31))

#add ticker symbol list
ticker_list=["AAPL","MSFT","GOOGL","META","TSLA","NVDA","ADBE","PYPL","INTC","CMCSA","NFLX","PEP"]
ticker=st.sidebar.selectbox('Company',ticker_list)

#fetch data from user inputs using yfinance library
data=yf.download(ticker,start=start_date,end=end_date)
#add date as column 
data.insert(0,"Date",data.index,True)
data.reset_index(drop=True,inplace=True)
st.write('Data from',start_date,'to',end_date)
st.write(data)

#plot the data
st.write("<p style='color:HotPink; font-size: 40px; font-family: 'Courier New';font-weight: bold;'>Data Visualization</p>",unsafe_allow_html=True)
st.write("<p style='color:lightPink; font-size: 25px; font-family: 'Courier New';font-weight: normal;'>Plot of the Data</p>",unsafe_allow_html=True)
fig=px.line(data,x='Date',y=data.columns,title='Closing price of the stock')
st.plotly_chart(fig)

#add a select box 
column=st.selectbox('Select column',data.columns[1:])

#subset of data
data=data[['Date',column]]
st.write('Selected Data')
st.write(data)

#ADF test check stationarity
st.write('Data Stationarity')
st.write(adfuller(data[column])[1]<0.05)

st.write("<p style='color:HotPink; font-size: 40px; font-family: 'Courier New';font-weight: bold;'>Decomposition of Data</p>",unsafe_allow_html=True)
decomposition=seasonal_decompose(data[column],model='additive',period=12)
st.write(decomposition.plot())

#make same plots in plotly
st.write('Evaluating Plots')
st.plotly_chart(px.line(x=data['Date'],y=decomposition.trend,title='Trend',width=800,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color='Blue'))
st.plotly_chart(px.line(x=data['Date'],y=decomposition.trend,title='Seasonality',width=800,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color='Green'))
st.plotly_chart(px.line(x=data['Date'],y=decomposition.trend,title='Residuals',width=800,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color='Red',line_dash='dot'))

p=st.slider('Select value of p',0,5,2)
d=st.slider('Select value of d',0,5,1)
q=st.slider('Select value of q',0,5,2)
seasonal_order=st.number_input('Select value of seasonal p',0,24,12)

model=sm.tsa.statespace.SARIMAX(data[column],order=(p,d,q),seasonal_order=(p,d,q,seasonal_order))
model=model.fit()

#print model summary
st.write("<p style='color:HotPink; font-size: 40px; font-family: 'Courier New';font-weight: bold;'>Model Summary</p>",unsafe_allow_html=True)
st.write(model.summary())
st.write('---')

#prediction model
st.write("<p style='color:HotPink; font-size: 40px; font-family: 'Courier New';font-weight: bold;'>Forecasting the Data</p>",unsafe_allow_html=True)
forecast_period=st.number_input('Select number of days for prediction',1,365,10)

predictions=model.get_prediction(start=len(data),end=len(data)+forecast_period-1)
predictions=predictions.predicted_mean

predictions.index=pd.date_range(start=end_date,periods=len(predictions),freq='D')
predictions=pd.DataFrame(predictions)
predictions.insert(0,'Date',predictions.index)

predictions.reset_index(drop=True,inplace=True)
st.write('Predictions',predictions)
st.write('Actual Data',data)
st.write('---')

fig=go.Figure()
fig.add_trace(go.Scatter(x=data['Date'],y=data[column],mode='lines',name='Actual',line=dict(color='blue')))
fig.add_trace(go.Scatter(x=predictions['Date'],y=predictions['predicted_mean'],mode='lines',name='Predicted',line=dict(color='red')))
fig.update_layout(title='Actual vs Predicted',xaxis_title='Date',yaxis_title='Price',width=800,height=400)
st.plotly_chart(fig)

#add buttons
show_plots=False
if st.button('Show separate plots'):
    if not show_plots:
        st.write(px.line(x=data['Date'],y=data[column],title='Actual',width=800,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color='Blue'))
        st.write(px.line(x=predictions['Date'],y=predictions['predicted_mean'],title='Predicted',width=800,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color='Red'))
        show_plots=True
    else:
        show_plots=False

hide_plots=False
if st.button('Hide separate plots'):
    if not hide_plots:
        hide_plots=True
    else:
        hide_plots=False



st.write("---")

st.write("<p style='color:HotPink ; font-size: 30px;font-family: 'Courier New'; font-weight: bold;'>About the Author</p>",unsafe_allow_html=True)
st.write("<p style='color:LightPink; font-size: 25px; font-family: 'Georgia';font-weight: bold;'>Shelly Bhalla</p>",unsafe_allow_html=True)

linkedin_url="https://www.linkedin.com/in/shelly-bhalla-58a7271b6"
github_url="https://github.com/Shellybhalla13"

linkedin_icon="https://static.vecteezy.com/system/resources/previews/017/339/624/original/linkedin-icon-free-png.png"
github_icon="https://cdn0.iconfinder.com/data/icons/shift-logotypes/32/Github-512.png"

st.markdown(f'<a href="{github_url}"><img src="{github_icon}" width="60" height="60"></a>'
            f'<a href="{linkedin_url}"><img src="{linkedin_icon}" width="60" height="60"></a>',unsafe_allow_html=True)