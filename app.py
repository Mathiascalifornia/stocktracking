########## Imports ##########
import pandas as pd , numpy as np
import plotly.express as px , plotly.graph_objects as go
import dash 
from dash import dcc 
from dash import html 
from dash.dependencies import Input, Output , State
import datetime as dt 
import talib as ta
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from pandas_datareader import data
import yfinance as yf
import fear_and_greed
import os
from threading import Timer
import base64
import warnings
from typing import Union , Iterable
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')
yf.pdr_override()



# To not display dash errors
import logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)
from flask import Flask  



class App:

    '''To hold the whole app'''
    def __init__(self):
        with open("bear_final.jpg", "rb") as image_file:
            self.bear = base64.b64encode(image_file.read())


        with open("bull.jpg", "rb") as image_file:
            self.bull = base64.b64encode(image_file.read())

        self.ticker_df_dict = {} # global variables
        self.ticker_pct_change = {}
        self.current_ticker = 'SP500'


        self.server = Flask(__name__)
        self.server.config['SECRET_KEY'] = os.urandom(24)
        self.server.config['SESSION_TYPE'] = 'filesystem'
        self.server.config['SESSION_PERMANENT'] = False
        self.app = dash.Dash(__name__, server=self.server)


        ########## Dashboard ##########
        self.app.layout = html.Div([
            dcc.Tabs(id='tabs', value='tab-1', children=[
                dcc.Tab(label='Input', value='tab-1', children=[
            
                    html.Div([
                        html.Img(src="data:image/png;base64,{}".format(self.bull.decode()), style={"height": "200px"}),
                        html.Div([
                            dcc.Markdown('## *Enter your tickers, separated by a space*', style={"width": "100%"}),
                            dcc.Input(
                                id='ticker-input',
                                type='text',
                                debounce=True,
                                placeholder='Enter a ticker...',
                                value='',
                                style={"width": "95%"}
                            ),
                            html.Button('START', id='submit-ticker', n_clicks=0, style={'width': '96%'})
                        ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'middle'}),
                        html.Img(src="data:image/png;base64,{}".format(self.bear.decode()), style={"height": "200px", "margin-left": "auto"})
                    ], style={
                        "display": "flex",
                        "flex-direction": "row",
                        "justify-content": "center",
                        "align-items": "center",
                        "text-align": "center",
                        "margin-top": "100px",
                        "margin-bottom": "100px"
                    })
                ]),
                dcc.Tab(label='Output', value='tab-2', children=[
                    html.Div(id='output')
                ])
            ])
        ])


    #def register_callbacks(self):
        @self.app.callback(
            Output(component_id='output', component_property='children'),
            Input('submit-ticker', 'n_clicks'),
            State('ticker-input', 'value')) # Allows to share data between elements
        def get_output(n_click , input_value):
            if n_click is None:
                raise dash.exceptions.PreventUpdate
            else:
                tickers = input_value.upper().split()

                if tickers != []:
                    
                    sp500 = data.get_data_yahoo('^GSPC' , dt.datetime(1975,1,1))
                    liste_stocks = get_data(tickers=tickers)
                    tickers.append('SP500')

                    ticker_pct_change = {}
                    ticker_df_dict = {}
                    liste_stocks.append(sp500)
                    pct_change_list = [get_pct_change(df) for df in liste_stocks]
                        
                    # To link the ticker and the dataframes
                    ticker_df_dict = dict(zip(tickers , liste_stocks))

                    for i in range(len(tickers)):
                        ticker_pct_change[tickers[i]] = pct_change_list[i]


                    self.ticker_df_dict = ticker_df_dict
                    self.ticker_pct_change = ticker_pct_change
                    
                    

                    return main(tickers , pct_change_list , ticker_pct_change , sp500 , liste_stocks , ticker_df_dict)


        # For the pct changes
        @self.app.callback(
        Output(component_id='pct_changes' , component_property='children'),
        Input(component_id='first_dropdown' , component_property='value') )
        def change_pct_changes(input_ticker):
            self.current_ticker = input_ticker
            return display_pct_changes(self.ticker_pct_change.get(input_ticker))


        # For the RSI figure
        @self.app.callback(
        Output(component_id='RSI' , component_property='figure'),
        Input(component_id='first_dropdown' , component_property='value'))
        def change_rsi_fig(input_ticker):
                if input_ticker:
                    return plot_rsi(self.ticker_df_dict.get(input_ticker))


        # For the volume plot
        @self.app.callback(
        Output(component_id='volume' , component_property='figure'),
        Input(component_id='first_dropdown' , component_property='value'))
        def change_volume_plot(input_ticker):
            if input_ticker:
                return plot_volume(self.ticker_df_dict.get(input_ticker))


        # For the bbands figure
        @self.app.callback(
        Output(component_id='bbands' , component_property='figure'),
        Input(component_id='first_dropdown' , component_property='value'))
        def change_bbands(input_ticker):
                if input_ticker:
                    return bbands(self.ticker_df_dict.get(input_ticker))
                            
        # For the adx figure
        @self.app.callback(
        Output(component_id='adx' , component_property='figure'),
        Input(component_id='first_dropdown' , component_property='value'))
        def change_bbands(input_ticker):
                if input_ticker:
                    return plot_candle_adx(self.ticker_df_dict.get(input_ticker))

        # For the macd figure
        @self.app.callback(
        Output(component_id='macd' , component_property='figure'),
        Input(component_id='first_dropdown' , component_property='value'))
        def change_macd(input_ticker):
                if input_ticker:
                    return plot_macd(self.ticker_df_dict.get(input_ticker))


        # For the seasonal figure
        @self.app.callback(
        Output(component_id='seasonal' , component_property='figure'),
        Input(component_id='first_dropdown' , component_property='value'))
        def change_seasonal(input_ticker):
                if input_ticker:
                    return get_the_three_season(self.ticker_df_dict.get(input_ticker))



        # For the titles
        @self.app.callback(
        Output(component_id='title' , component_property='children'),
        Input(component_id='first_dropdown' , component_property='value'))
        def change_title(input_ticker):
                if input_ticker:
                    return f'RSI , Bollinger bands , ADX , MACD , trading volume and analysis of seasonality for {input_ticker} :'
                            


        # For the percentage change title
        @self.app.callback(
        Output(component_id='title_percentage' , component_property='children'),
        Input(component_id='first_dropdown' , component_property='value'))
        def change_pct_title(input_ticker):
                if input_ticker:
                    return f'Changes in percentage for {input_ticker} :' 



    # For deployement :(
    """
    def run(self):
        if __name__ == '__main__':
            self.app.run_server(host='0.0.0.0', port=int(os.environ.get('PORT', 8050))) # To find the heroku environment variable PORT  
    """

    def run(self):
        def open_browser():
            import webbrowser
            if not os.environ.get("WERKZEUG_RUN_MAIN"):
                webbrowser.open_new('http://127.0.0.1:8050/')

        if __name__ == "__main__":
            Timer(1, open_browser).start()
            self.app.run_server()



##### Functions #####
def get_data(tickers : Iterable) -> list:
    ''' 
    Get the data from yahoo finance , using the list of tickers
    '''
    return [data.get_data_yahoo(tick , dt.datetime(1975,1,1)) for tick in tickers]




def get_pct_change(df : pd.DataFrame) -> dict:
    """ 
    Compute the difference in percentage for several 
    period of time , returns a dict with the time period as key
    and percentage change as value 
    """
    current = float(df.iloc[-1]['Adj Close'])


    # Week
    previous , week_diff = prev_diff(7 , current , df) \
                        or prev_diff(5, current , df) \
                        or prev_diff(6, current , df) \
                        or prev_diff(9, current , df) \
                        or "N/A"

    # Two weeks
    previous , twoweek_diff = prev_diff(14, current , df) \
                            or prev_diff(10, current , df) \
                            or prev_diff(16, current , df) \
                            or prev_diff(12, current , df) \
                            or "N/A"       

    # Six months
    previous , sixmonth_diff = prev_diff(180, current , df) \
                            or prev_diff(6*19, current , df) \
                            or prev_diff(6*20, current , df) \
                            or prev_diff(6*21, current , df) \
                            or  prev_diff(6*22, current , df) \
                            or "N/A"

    # One year
    previous , oneyear_diff = prev_diff(252, current , df) \
                            or prev_diff(251, current , df) \
                            or prev_diff(250, current , df) \
                            or prev_diff(249, current , df) \
                            or prev_diff(253, current , df) \
                            or "N/A"

    # Five years
    previous , fiveyear_diff = prev_diff(365*5, current , df) \
                            or prev_diff(252*5, current , df) \
                            or prev_diff(251*5, current , df) \
                            or prev_diff(250*5, current , df) \
                            or prev_diff(249*5, current , df) \
                            or prev_diff(253*5, current , df) \
                            or "N/A"


    # Overall
    previous = float(df.iloc[0]['Adj Close'])
    overall_diff = np.round(100*((current - previous) / previous), 2)
    
    # Return the dictionnary with all the diff
    return {'One week : ' : week_diff ,
            'Two weeks : ' : twoweek_diff ,
            'Six months : ' : sixmonth_diff ,
            'One year : ' : oneyear_diff ,
            'Five years : ' : fiveyear_diff ,
            'Overall : ' : overall_diff}



def minmax_scale(days : int , listestock : Iterable) -> pd.DataFrame:
    ''' Scale the data , to make comparaison possible between stocks '''
    period_df = []
    for df in listestock:
        df_ = df[df.index[-1] - dt.timedelta(days):]
        df_['normalized'] = MinMaxScaler().fit_transform(df_['Adj Close'].values.reshape(-1,1))
        period_df.append(df_)
    return period_df



def display_pct_changes(ticker_diff_dict : dict) -> str: 
    ''' Display the change in percentage already created '''
    string = f'   '
    for key , val in ticker_diff_dict.items():
        string += key
        if isinstance(val , np.number):
            if val >= 0:
                string += '+'

        string += str(val)
        string += ' %'
        string += ' | '

    #fg = fear_and_greed.get()
    return string


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


def spaces(n=3) -> html.Div: 
    """ 
    To avoid repetition , create as many spaces as you specify in "n"
    """
    return html.Div([html.Br() for i in range(n)])

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



def add_a_trace(liste , fig , tickers):
    ''' Add a trace in the figure'''
    for i in range(len(liste)):
        fig.add_trace(go.Scatter(y=liste[i]['normalized'] , hovertext=liste[i]['Adj Close'].apply(lambda x : str(f'Price : {np.round(x,2)}')), x=liste[i].index , name=tickers[i] , mode='lines', visible=False))



def prev_diff(n  , current , df) -> Union[tuple , None]:
    '''Return previous and diff , or None in case of error'''
    try:
        previous = float(df[df.index == str(df.index[-1] - dt.timedelta(n))]['Adj Close'])
        diff = np.round(100*((current - previous) / previous),2)
        return previous , diff
    except:
            return None



def prepare_for_decompose(df : pd.DataFrame) -> pd.DataFrame:
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

def get_the_three_season(df : pd.DataFrame) -> go.Figure:
     
    to_dec = prepare_for_decompose(df)
    fig_yearly = plot_seasonality(to_plot=to_dec , title='yearly' , hover_data='month')
    
    fig_yearly.update_layout(width=1470 , height=700)


    return fig_yearly



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



def main(tickers , pct_change_list , ticker_pct_change , sp500 , liste_stocks , ticker_df_dict):



    liste_stock_overall = minmax_scale(365*47 , liste_stocks)
    liste_two_week = minmax_scale(14 , liste_stocks)
    liste_six_month = minmax_scale(30*6 , liste_stocks)
    liste_one_year = minmax_scale(365 , liste_stocks)
    liste_five_year = minmax_scale(365*5 , liste_stocks)



    # Function to add the traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=liste_stock_overall[-1]['normalized'] , hovertext=liste_stock_overall[-1]['Adj Close'].apply(lambda x : str(f'Price : {np.round(x,2)}')), x=liste_stock_overall[-1].index , name=tickers[-1]  , mode='lines' , visible=True))
    for i in range(len(liste_stocks) -1):
        fig.add_trace(go.Scatter(y=liste_stock_overall[i]['normalized'] , hovertext=liste_stock_overall[i]['Adj Close'].apply(lambda x : str(f'Price : {np.round(x,2)}')), x=liste_stock_overall[i].index , name=tickers[i] , mode='lines', visible='legendonly'))


    # Add the traces
    add_a_trace(liste_two_week , fig=fig , tickers=tickers)
    add_a_trace(liste_six_month , fig=fig , tickers=tickers)
    add_a_trace(liste_one_year , fig=fig , tickers=tickers)
    add_a_trace(liste_five_year , fig=fig , tickers=tickers)


    liste1 = [*['legendonly' for i in range(len(tickers) - 1)] , True  , *[False for i in range(len(tickers)*4)]]
    liste2 = [*[False for i in range(len(tickers))] , *['legendonly' for i in range(len(tickers) - 1)] , True , *[False for i in range(len(tickers)*3)]]
    liste3 = [*[False for i in range(len(tickers)*2)] , *['legendonly' for i in range(len(tickers) - 1)] , True ,  *[False for i in range(len(tickers)*2)]]
    liste4 = [*[False for i in range(len(tickers)*3)] , *['legendonly' for i in range(len(tickers) - 1)] , True , *[False for i in range(len(tickers))]]
    liste5 = [*[False for i in range(len(tickers)*4)] , *['legendonly' for i in range(len(tickers) - 1)] , True]

    # Create the buttons
    dropdown_buttons = [
    {'label': "ALL", 'method': "update", 'args': [{"visible": liste1} , {'title' : 'Overall normalized stock prices'}] },
    {'label': "2WTD", 'method': "update", 'args': [{"visible": liste2} , {'title' : 'Two weeks normalized stock prices'}]},
    {'label': "6MTD", 'method': "update", 'args': [{"visible": liste3} , {'title' : 'Six months normalized stock prices'}]},
    {'label': "1YTD", 'method': "update", 'args': [{"visible": liste4} , {'title' : 'One year normalized stock prices'}]},
    {'label': "5YTD", 'method': "update", 'args': [{"visible":liste5} , {'title' : 'Five years normalized stock prices'}]}
    ]

    # Update the figure to add dropdown menu
    fig.update_layout({
            'updatemenus': [
                {'type': "dropdown",
                'showactive': True,'active': 0, 'buttons' : dropdown_buttons},
            ]})

    # To see all the prices 
    fig.update_layout(hovermode = 'x unified')

    # The figsize
    fig.update_layout(autosize=False , width=1450 , height=500)

    # The title
    fig.update_layout(
        title={
            'text': "Share price over time",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})




    # First div
    app.layout = html.Div(id='Price' , children=[
         
    # Main title
    html.H1(children=['---------------------------------------------------------  Stocks tracking  ---------------------------------------------------------'] , style={'border' : '1px solid black'}),

    # Main graph
    dcc.Graph(figure=fig),

    # Fear and greed index
    dcc.Graph(id='f&g' , figure=plot_fear_and_greed()),

    # Add the dropdown for the rsi and plain text (dropdown to the left , plain text to the right , rsi below)
    dcc.Dropdown(id='first_dropdown',
    options= [{'label' : tick , 'value' : tick} for tick in tickers] , value=tickers[-1]),


    html.Br(), # Jump a line

    # Title for the percentages change
    html.H3(children = ['Changes in percentage for SP500 :'] , id='title_percentage') ,

    html.Br(), # Jump a line 


    # Plain text
    html.I(children=[display_pct_changes(pct_change_list[-1])] , style={'border':'1px solid black'} , id='pct_changes'),


    html.Br(),
    html.Br(),

    # Title for the RSI fig
    html.H3(children=[f'RSI , Bollinger bands , ADX , MACD , trading volume and analysis of seasonality for SP500 :'] , id='title'),

    # Figures
    dcc.Graph(id='RSI' , figure=plot_rsi(sp500)),
    dcc.Graph(id='bbands' , figure=bbands(sp500)),
    dcc.Graph(id='adx' , figure=plot_candle_adx(sp500)),
    dcc.Graph(id='macd' , figure=plot_macd(sp500)),
    dcc.Graph(id='volume' , figure=plot_volume(sp500)),
    dcc.Graph(id='seasonal' , figure=get_the_three_season(df=sp500)),

    ] , style={'text-align' : 'center'})


    return app.layout 









##### Launch the app #####
app = App()
app.run()
print(input('Tap anywhere to close'))

