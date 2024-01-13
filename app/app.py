
import os
from threading import Timer
import warnings
import base64
import datetime as dt 
import logging

import numpy as np
import plotly.graph_objects as go

import dash 
from dash import dcc 
from dash import html 
from dash.dependencies import Input, Output , State
from flask import Flask  

from pandas_datareader import data
import yfinance as yf

warnings.filterwarnings('ignore')
yf.pdr_override()

# To not display dash errors
logging.getLogger('werkzeug').setLevel(logging.ERROR)


from utils.utils import Utils
from utils.viz import Viz


class App:

    utils = Utils()
    viz = Viz()

    ressource_path = os.path.join(os.path.dirname(os.path.dirname(__file__)) , "ressources")
    bear_path = os.path.join(ressource_path , "bear.jpg")
    bull_path = os.path.join(ressource_path ,  "bull.jpg")

    def __init__(self):


        self.bear = App._load_image_as_bytes(App.bear_path)
        self.bull = App._load_image_as_bytes(App.bull_path)


        self.ticker_df_dict = {} 
        self.ticker_pct_change = {}
        self.current_ticker = 'SP500'


        self.server = App._load_server()
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
                    liste_stocks = App.utils.get_data(tickers=tickers)
                    tickers.append('SP500')

                    ticker_pct_change = {}
                    ticker_df_dict = {}
                    liste_stocks.append(sp500)
                    pct_change_list = [App.utils.get_pct_change(df) for df in liste_stocks]
                        
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
            return App.utils.display_pct_changes(self.ticker_pct_change.get(input_ticker))


        # For the RSI figure
        @self.app.callback(
        Output(component_id='RSI' , component_property='figure'),
        Input(component_id='first_dropdown' , component_property='value'))
        def change_rsi_fig(input_ticker):
                if input_ticker:
                    return App.viz.plot_rsi(self.ticker_df_dict.get(input_ticker))


        # For the volume plot
        @self.app.callback(
        Output(component_id='volume' , component_property='figure'),
        Input(component_id='first_dropdown' , component_property='value'))
        def change_volume_plot(input_ticker):
            if input_ticker:
                return App.viz.plot_volume(self.ticker_df_dict.get(input_ticker))


        # For the bbands figure
        @self.app.callback(
        Output(component_id='bbands' , component_property='figure'),
        Input(component_id='first_dropdown' , component_property='value'))
        def change_bbands(input_ticker):
                if input_ticker:
                    return App.viz.bbands(self.ticker_df_dict.get(input_ticker))
                            
        # For the adx figure
        @self.app.callback(
        Output(component_id='adx' , component_property='figure'),
        Input(component_id='first_dropdown' , component_property='value'))
        def change_bbands(input_ticker):
                if input_ticker:
                    return App.viz.plot_candle_adx(self.ticker_df_dict.get(input_ticker))

        # For the macd figure
        @self.app.callback(
        Output(component_id='macd' , component_property='figure'),
        Input(component_id='first_dropdown' , component_property='value'))
        def change_macd(input_ticker):
                if input_ticker:
                    return App.viz.plot_macd(self.ticker_df_dict.get(input_ticker))


        # For the seasonal figure
        @self.app.callback(
        Output(component_id='seasonal' , component_property='figure'),
        Input(component_id='first_dropdown' , component_property='value'))
        def change_seasonal(input_ticker):
                if input_ticker:
                    return App.viz.get_the_three_season(self.ticker_df_dict.get(input_ticker))



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



    def run(self):
        def open_browser():
            import webbrowser
            if not os.environ.get("WERKZEUG_RUN_MAIN"):
                webbrowser.open_new('http://127.0.0.1:8050/')

        if __name__ == "__main__":
            Timer(1, open_browser).start()
            self.app.run_server()




    @staticmethod
    def _load_image_as_bytes(image_path:str) -> bytes:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read())
    
    @staticmethod
    def _load_server() -> Flask:
        server = Flask(__name__)
        server.config['SECRET_KEY'] = os.urandom(24)
        server.config['SESSION_TYPE'] = 'filesystem'
        server.config['SESSION_PERMANENT'] = False
        return server


    @staticmethod
    def spaces(n=3) -> html.Div: 
        """ 
        To avoid repetition , create as many spaces as you specify in "n"
        """
        return html.Div([html.Br() for i in range(n)])



def main(tickers , pct_change_list , ticker_pct_change , sp500 , liste_stocks , ticker_df_dict):



    liste_stock_overall = App.utils.minmax_scale(365*47 , liste_stocks)
    liste_two_week = App.utils.minmax_scale(14 , liste_stocks)
    liste_six_month = App.utils.minmax_scale(30*6 , liste_stocks)
    liste_one_year = App.utils.minmax_scale(365 , liste_stocks)
    liste_five_year = App.utils.minmax_scale(365*5 , liste_stocks)



    # Function to add the traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=liste_stock_overall[-1]['normalized'] , hovertext=liste_stock_overall[-1]['Adj Close'].apply(lambda x : str(f'Price : {np.round(x,2)}')), x=liste_stock_overall[-1].index , name=tickers[-1]  , mode='lines' , visible=True))
    for i in range(len(liste_stocks) -1):
        fig.add_trace(go.Scatter(y=liste_stock_overall[i]['normalized'] , hovertext=liste_stock_overall[i]['Adj Close'].apply(lambda x : str(f'Price : {np.round(x,2)}')), x=liste_stock_overall[i].index , name=tickers[i] , mode='lines', visible='legendonly'))


    # Add the traces
    App.viz.add_a_trace(liste_two_week , fig=fig , tickers=tickers)
    App.viz.add_a_trace(liste_six_month , fig=fig , tickers=tickers)
    App.viz.add_a_trace(liste_one_year , fig=fig , tickers=tickers)
    App.viz.add_a_trace(liste_five_year , fig=fig , tickers=tickers)


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
    dcc.Graph(id='f&g' , figure=App.viz.plot_fear_and_greed()),

    # Add the dropdown for the rsi and plain text (dropdown to the left , plain text to the right , rsi below)
    dcc.Dropdown(id='first_dropdown',
    options= [{'label' : tick , 'value' : tick} for tick in tickers] , value=tickers[-1]),


    html.Br(), # Jump a line

    # Title for the percentages change
    html.H3(children = ['Changes in percentage for SP500 :'] , id='title_percentage') ,

    html.Br(), # Jump a line 


    # Plain text
    html.I(children=[App.utils.display_pct_changes(pct_change_list[-1])] , style={'border':'1px solid black'} , id='pct_changes'),


    html.Br(),
    html.Br(),

    # Title for the RSI fig
    html.H3(children=[f'RSI , Bollinger bands , ADX , MACD , trading volume and analysis of seasonality for SP500 :'] , id='title'),

    # Figures
    dcc.Graph(id='RSI' , figure=App.viz.plot_rsi(sp500)),
    dcc.Graph(id='bbands' , figure=App.viz.bbands(sp500)),
    dcc.Graph(id='adx' , figure=App.viz.plot_candle_adx(sp500)),
    dcc.Graph(id='macd' , figure=App.viz.plot_macd(sp500)),
    dcc.Graph(id='volume' , figure=App.viz.plot_volume(sp500)),
    dcc.Graph(id='seasonal' , figure=App.viz.get_the_three_season(df=sp500)),

    ] , style={'text-align' : 'center'})


    return app.layout 


##### Launch the app #####
app = App()
app.run()
print(input('Tap anywhere to close'))
