########## Imports ##########
import datetime as dt 
import warnings

import pandas as pd , numpy as np

import plotly.express as px , plotly.graph_objects as go

import talib as ta
import yfinance as yf
import fear_and_greed
from statsmodels.tsa.seasonal import seasonal_decompose

from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')
yf.pdr_override()


class Viz:

    @staticmethod
    def plot_rsi(df : pd.DataFrame) -> go.Figure:
        """ 
        Plot the Relative Strenght Index figure 
        """
        df['RSI'] = ta.RSI(df['Adj Close'])

        # We'll do that with simple buttons
        df['RSI'] = ta.RSI(df['Adj Close'])


        figure = px.line(df[df.index[-1] - dt.timedelta(252*5) :]['RSI'])
        figure.add_hline(y=70 , line_color='red')
        figure.add_hline(y=30  ,line_color='green')

        buttons = [
        {'count': 15, 'label': "3WTD", 'step': 'day', 'stepmode': 'todate'},
        {'count': 21*6, 'label': "6MTD", 'step': 'day', 'stepmode': 'todate'},
        {'count': 252*5, 'label': "5YTD", 'step': 'day', 'stepmode': 'todate'}
        ]

        figure.update_layout({'xaxis' : {'rangeselector' : {'buttons' : buttons}}})
        figure.update_layout(width=1420 , height=500)
        figure.update_layout(
                    title={
                        'text': "RSI",
                        'y':0.99,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'})
        figure.update_layout(xaxis_title='')
        return figure


    @staticmethod
    def plot_fear_and_greed() -> go.Figure:
        '''Plot the fear and greed index'''

        description = fear_and_greed.get()[1]
        value_ = fear_and_greed.get()[0]

        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = value_,
            title = {'text': f"Fear and Greed index ({description})"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {'axis': {'range': [None, 100]},
                        'steps' : [{'range': [50, 100], 'color': "lightgray"}]}
        ))

        return fig



    @staticmethod
    def bbands(df : pd.DataFrame) -> go.Figure: 
        ''' Plot the bollinger bands figure '''   
        df_ = df[(df.index[-1] - dt.timedelta(365*5)):]

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df_['Adj Close'] , x=df_.index , mode='lines' , fillcolor='blue' , name='Price'))

        # Define the Bollinger Bands with 2-sd
        upper_2sd, mid_2sd, lower_2sd = ta.BBANDS(df_['Adj Close'],
                                                    nbdevup=2,
                                                    nbdevdn=2,
                                                    timeperiod=20)

        upper_2sd = pd.DataFrame(upper_2sd)
        mid_2sd = pd.DataFrame(mid_2sd)
        lower_2sd = pd.DataFrame(lower_2sd)

        fig.add_trace(go.Scatter(x=upper_2sd.index , y=upper_2sd[0] , line=dict(color='rgb(255,0,0)') , name='Upper band') )
        fig.add_trace(go.Scatter(x=mid_2sd.index , y=mid_2sd[0] , line=dict(color='rgb(0,255,0)') , name='SMA 20'))
        fig.add_trace(go.Scatter(x=lower_2sd.index , y=lower_2sd[0] , line=dict(color='rgb(255,0,0)') , name='Lower band'))

        buttons = [
        {'count': 31, 'label': "1MTD", 'step': 'day', 'stepmode': 'todate'},
        {'count' : 365 , 'label' : '1YTD' , 'step' : 'day' , 'stepmode' : 'todate'},
        {'count' : 365*5 , 'label' : '5YTD' , 'step' : 'day' , 'stepmode' : 'todate'}
            ]

        fig.update_layout({'xaxis' : {'rangeselector' : {'buttons' : buttons}}})
        fig.update_layout(hovermode = 'x unified')
        fig.update_layout(width=1465 , height=500)
        fig.update_layout(
                    title={
                        'text': "Bollinger Bands",
                        'y':0.9,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'})

        return fig


    @staticmethod
    def plot_volume(df : pd.DataFrame) -> go.Figure:
        ''' Plot the volume figure '''
        vol_fig = px.bar(df , x=df.index , y=df['Volume'] , color_discrete_sequence=['white'])
        buttons = [
            {'count': 15, 'label': "3WTD", 'step': 'day', 'stepmode': 'todate'},
            {'count': 21*6, 'label': "6MTD", 'step': 'day', 'stepmode': 'todate'},
            {'count': 252*5, 'label': "5YTD", 'step': 'day', 'stepmode': 'todate'},
            {'count': 252*47, 'label': "ALL", 'step': 'day', 'stepmode': 'todate'}]

        vol_fig.update_layout(width=1395 , height=500 , paper_bgcolor="white", plot_bgcolor="black")
        vol_fig.update_layout({'xaxis' : {'rangeselector' : {'buttons' : buttons}}})

        vol_fig.update_layout(
                    title={
                        'text': "Volume",
                        'y':0.95,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'})
        vol_fig.update_layout(xaxis_title='')
        return vol_fig

    @staticmethod
    def add_a_trace(liste , fig , tickers):
        ''' Add a trace in the figure'''
        for i in range(len(liste)):
            fig.add_trace(go.Scatter(y=liste[i]['normalized'] , hovertext=liste[i]['Adj Close'].apply(lambda x : str(f'Price : {np.round(x,2)}')), x=liste[i].index , name=tickers[i] , mode='lines', visible=False))



    @staticmethod
    def plot_seasonality(to_plot : pd.DataFrame , title : str , hover_data : str) -> go.Figure:
        """ 
        Plot seasonality figure
        """
        
        fig = make_subplots(rows=3, cols=1 ,  shared_xaxes=True)

        observed = px.line(data_frame=to_plot , x='date', y='observed' , hover_data=[hover_data] , color_discrete_sequence=['blue'])
        trend = px.line(data_frame=to_plot , x='date', y='trend' , hover_data=[hover_data] ,  color_discrete_sequence=['purple'])
        season = px.line(data_frame=to_plot , x='date', y='seasonal' , hover_data=[hover_data], color_discrete_sequence=['red'])



        fig.add_trace(observed.data[0] , row=1, col=1)
        fig.add_trace(trend.data[0]  , row=2, col=1)
        fig.add_trace(season.data[0] , row=3, col=1)



        def add_custom_legend(color : str , title : str):

            return fig.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=[color],
                        showscale=False
                    ),
                    showlegend=True,
                    name=title,
                ))


        add_custom_legend('blue' , 'Observed')
        add_custom_legend('purple' , 'Trend')
        add_custom_legend('red' , 'Seasonality')

        fig.update_layout(
            title={
                'text': f"Breakdown of annual seasonality",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

        return fig

    @staticmethod
    def get_the_three_season(df : pd.DataFrame) -> go.Figure:
        

        def __prepare_for_decompose(df : pd.DataFrame) -> pd.DataFrame:
            """ 
            Prepare data for the "plot_seasonality" method
            """
            to_dec_yearly = df[df.index[-1] - dt.timedelta(252*7):] # for yearly
            to_dec_yearly['month'] = to_dec_yearly.index.month_name()
            decomposition_yearly = seasonal_decompose(to_dec_yearly['Adj Close'], model='additive', period=252)
            to_plot_yearly = pd.DataFrame({'seasonal' : decomposition_yearly.seasonal.values , 'trend' : decomposition_yearly.trend.values ,
                                        'month' : to_dec_yearly['month'] , 'date' : decomposition_yearly.seasonal.index ,
                                        'observed' : decomposition_yearly.observed})
            
            return to_plot_yearly
            
        to_dec = __prepare_for_decompose(df)
        fig_yearly = Viz.plot_seasonality(to_plot=to_dec , title='yearly' , hover_data='month')

        fig_yearly.update_layout(width=1470 , height=700)


        return fig_yearly


    @staticmethod
    def plot_macd(df : pd.DataFrame) -> go.Figure: 

        to_plot = df[df.index[-1] - dt.timedelta(252*4):]

        # Create trace for  Close
        trace_to_plot_close = go.Scatter(x=to_plot.index, y=to_plot['Close'], name='Close')

        # Create trace for MACD line
        to_plot['26ema'] = to_plot['Close'].ewm(span=26).mean()
        to_plot['12ema'] = to_plot['Close'].ewm(span=12).mean()
        to_plot['macd'] = to_plot['12ema'] - to_plot['26ema']
        trace_macd = go.Scatter(x=to_plot.index, y=to_plot['macd'], name='MACD',mode='lines')

        # Create trace for signal line
        to_plot['signal'] = to_plot['macd'].ewm(span=9).mean()
        trace_signal = go.Scatter(x=to_plot.index, y=to_plot['signal'], name='Signal',mode='lines')

        # Create trace for histogram
        to_plot['hist'] = to_plot['macd'] - to_plot['signal']
        colors = np.array(['green' if x>0 else 'red' for x in to_plot['hist']])
        trace_hist = go.Bar(x=to_plot.index, y=to_plot['hist'], name='Histogram',marker=dict(color=colors))

        # Create subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

        # Add trace to subplot
        fig.add_trace(trace_to_plot_close, row=1, col=1)
        fig.add_trace(trace_macd, row=2, col=1)
        fig.add_trace(trace_signal, row=2, col=1)
        fig.add_trace(trace_hist, row=2, col=1)

        # Update layout
        fig.update_layout(title='Close and MACD', xaxis=dict(title='Date'), yaxis=dict(title='Price'))
        fig.update_layout(xaxis=dict(showticklabels=False))


        fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([

                            dict(count=3,
                                label="3YTD",
                                step="year",
                                stepmode="todate"
                                ),


                            dict(count=2,
                                label="2YTD",
                                step="year",
                                stepmode="todate"
                                ),

                            dict(count=6,
                                label="6MTD",
                                step="month",
                                stepmode="todate"),

                            dict(count=15,
                                label="3WTD",
                                step="day",
                                stepmode="todate"),
                            

                        ])
                    ),
                    rangeslider=dict(
                        visible=False
                    ),
                    type="date"
                    
                        )
                    )



        fig.update_layout(width=1460 , height=700)
        fig.update_layout(
                    title={
                        'text': "Price and Moving Average Convergence Divergence",
                        'y':0.95,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'})

        fig.update_layout(xaxis_title='')
        return fig

    @staticmethod
    def plot_candle_adx(df : pd.DataFrame) -> go.Figure:
        ''' Plot candlesticks of the last 12 years with the ADX , to assess the trend '''

        df['ADX'] = ta.ADX(df['High'] , df['Low'] , df['Close'] , timeperiod=14)
        df_ = df.copy()

        time_periods = [252*12, 252*10, 252*9, 252*8 , 252*7 , 252*6 , 252*5, 252*2, 252*3, 252*2, 252 , 125 , 75]

        for period in time_periods:
            try:
                df_ = df_[df.index[-1] - dt.timedelta(period):]
                break  # Get out of the loop if there is no mistakes
            except:
                continue  # Try the next value if the time period don't exist

        fig = make_subplots(rows=2, cols=1 ,  shared_xaxes=True)
        fig.add_trace(go.Candlestick(x=df_.index,
                    open=df_['Open'], high=df_['High'],
                    low=df_['Low'], close=df_['Close'] , name='Candles') , row=1 , col=1) 
        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.add_trace(go.Scatter(x=df_.index , y=df_['ADX'] , line_color='blue' , name='ADX'), row=2 , col=1)
        fig.add_hline(y=25 , line_color='red' , row=2)

        fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([

                    dict(count=6,
                        label="6MTD",
                        step="month",
                        stepmode="todate"),

                    dict(count=2,
                        label="2YTD",
                        step="year",
                        stepmode="todate"
                        ),

                    dict(count=5,
                        label="5YTD",
                        step="year",
                        stepmode="todate"),
                    
                    dict(count=10,
                        label="10YTD",
                        step="year",
                        stepmode="todate"),

                ])
            ),
            rangeslider=dict(
                visible=False
            ),
            type="date"
            
                )
            )

        fig.update_layout(xaxis=dict(showticklabels=False))
        fig.update_layout(width=1442 , height=700)
        fig.update_layout(
                    title={
                        'text': "Prices and ADX",
                        'y':0.95,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'})
        fig.update_layout(xaxis_title='')
        return fig
