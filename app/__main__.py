
import os
from threading import Timer
import warnings
import logging
from typing import List , Iterable, Literal


import pandas as pd
import webbrowser
import numpy as np
import plotly.graph_objects as go

import dash 
from dash import dcc 
from dash import html 
from dash.dependencies import Input, Output , State
from flask import Flask  

import yfinance as yf

warnings.filterwarnings('ignore')
yf.pdr_override()

# To not display dash errors
logging.getLogger('werkzeug').setLevel(logging.ERROR)


from utils.utils import Utils
from utils.viz import Viz

# TODO ; Export the first fig time spans approach to all the figs, using the data_preprocessor script


class App:

    utils = Utils() # All utils functions
    viz = Viz() # All data viz related functions

    # Add tickers if you want them to be displayed by default (with the clean name as key and the actual ticker as value)
    tickers_to_fetch_by_default = {"SP500" : '^GSPC' , 
                                   "EUR:USD":'EURUSD=X'}

    # The bear and bull that will be displayed on the first screen
    ressource_path = os.path.join(os.path.dirname(os.path.dirname(__file__)) , "ressources")
    bear_path = os.path.join(ressource_path , "bear.jpg")
    bull_path = os.path.join(ressource_path ,  "bull.jpg")


    sector_compositions = {"Materials" : ("^SP500-15",) , 
                           "Energy" : ("^GSPE",) , 
                           "Financials" : ("^SP500-40",) , 
                           "Industrials" : ("^SP500-20",) , 
                           "Utilities" : ("^SP500-55",) , 
                           "Consumer Staples" : ("^SP500-30",) , 
                           "Consumer Discretionary" : ("^SP500-25",) , 
                           "Health Care" : ("^SP500-35",) , 
                           "Information Technology" : ("^SP500-45",) , 
                           "Communication Services" : ("^SP500-50",) , 
                           "Real Estate" : ("^SP500-60",)}

    def __init__(self):

        self.to_add_by_default_data:List[pd.DataFrame]
        self.to_add_by_default_data = App.utils.get_data(tickers=App.tickers_to_fetch_by_default.values())


        self.bear:bytes = App.utils.load_image_as_bytes(App.bear_path)
        self.bull:bytes = App.utils.load_image_as_bytes(App.bull_path)

        self.benchmarks_sectors:List[pd.DataFrame] = App.utils._create_benchmarks_sectors(App.sector_compositions)

        self.ticker_df_dict = {} 
        self.ticker_pct_change = {}
        self.current_ticker = 'SP500'

        self.server:Flask
        self.server = App.utils._load_server()
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

                tickers:list = input_value.upper().split()

                if tickers:
                    
                    liste_stocks = App.utils.get_data(tickers=tickers)
                    tickers.extend(list(App.sector_compositions.keys()))
                    tickers.extend(list(App.tickers_to_fetch_by_default)[::-1]) # Reverse the list to make the SP500 always appear first
                    

                    ticker_pct_change = {}
                    ticker_df_dict = {}


                    liste_stocks.extend(self.benchmarks_sectors)
                    liste_stocks.extend(self.to_add_by_default_data[::-1]) # Reverse the list to make the SP500 always appear first
                    

                    assert all(isinstance(el , (pd.DataFrame , pd.Series)) for el in liste_stocks) , "Bad elements in liste_stocks"

                    pct_change_list = [App.utils.get_pct_change(df) for df in liste_stocks]
                        
                    # To link the ticker and the dataframes
                    ticker_df_dict = dict(zip(tickers , liste_stocks))

                    for i in range(len(tickers)):
                        ticker_pct_change[tickers[i]] = pct_change_list[i]

                    self.ticker_df_dict = ticker_df_dict
                    self.ticker_pct_change = ticker_pct_change
                    
                    return App.main(tickers , pct_change_list , 
                                    self.to_add_by_default_data[0] , # SP500 will always be the first element
                                    liste_stocks)
                
                if not tickers:

                    return html.Div(
                                html.H2("You need to provide at least one valid ticker. Refer to the yahoo finance site for the available tickers"),
                                style={'text-align': 'center'}
                                    )
                     


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


    @staticmethod
    def main(tickers:Iterable , pct_change_list:Iterable ,
             sp500:pd.DataFrame , liste_stocks:Iterable) -> html.Div:

        liste_stock_overall:list[pd.DataFrame] = App.utils.minmax_scale(365*47 , liste_stocks)
        liste_two_week:list[pd.DataFrame] = App.utils.minmax_scale(14 , liste_stocks)
        liste_one_month:list[pd.DataFrame] = App.utils.minmax_scale(30 , liste_stocks)
        liste_six_month:list[pd.DataFrame] = App.utils.minmax_scale(30*6 , liste_stocks)
        liste_one_year:list[pd.DataFrame] = App.utils.minmax_scale(365 , liste_stocks)
        liste_five_year:list[pd.DataFrame] = App.utils.minmax_scale(365*5 , liste_stocks)


        fig = go.Figure()

        # # The "by default" status of the fig
        fig.add_trace(go.Scatter(y=liste_stock_overall[-1]['normalized'] , hovertext=liste_stock_overall[-1]['Adj Close'].apply(lambda x : str(f'Price : {np.round(x,2)}')), x=liste_stock_overall[-1].index , name=tickers[-1]  , mode='lines' , visible=True))
        
        for i in range(len(liste_stocks)-1):
            fig.add_trace(go.Scatter(y=liste_stock_overall[i]['normalized'] , hovertext=liste_stock_overall[i]['Adj Close'].apply(lambda x : str(f'Price : {np.round(x,2)}')), x=liste_stock_overall[i].index , name=tickers[i] , mode='lines', visible='legendonly'))


        # Add the traces
        for list_ in (liste_two_week, liste_one_month,  liste_six_month, liste_one_year, liste_five_year):
            App.viz.add_a_trace(list_ , fig=fig , tickers=tickers)


        labels = (
            "2WTD",
            "1MTD",
            "6MTD",
            "1YTD",
            "5YTD"
        )

        n_time_period = len(labels)

        first_n_false = 1
        last_n_false = n_time_period-1

        legend_only_templ = ['legendonly' for i in range(len(tickers) - 1)]

        results = {
            
                "ALL" : [ 
                    *['legendonly' for i in range(len(tickers) - 1)],
                    *[False for i in range(len(tickers)*5)],
                    True,
                ] , 
                

                }

        for label in labels:

            
            result = [ 

                *[False for i in range(len(tickers) * first_n_false)] , 
                *legend_only_templ, 
                True,
                *[False for i in range(len(tickers) * last_n_false)]
            ]
            

            first_n_false += 1
            last_n_false -= 1

            

            results[label] = result


        # Create the buttons
        dropdown_buttons = [
        {'label': "ALL", 'method': "update", 'args': [{"visible": results["ALL"]} , {'title' : 'Overall normalized stock prices'}]},
        {'label': "2WTD", 'method': "update", 'args': [{"visible": results["2WTD"]} , {'title' : 'Two weeks normalized stock prices'}]},
        {'label': "1MTD", 'method': "update", 'args': [{"visible": results["1MTD"]} , {'title' : 'One month normalized stock prices'}]},
        {'label': "6MTD", 'method': "update", 'args': [{"visible": results["6MTD"]} , {'title' : 'Six months normalized stock prices'}]},
        {'label': "1YTD", 'method': "update", 'args': [{"visible": results["1YTD"]} , {'title' : 'One year normalized stock prices'}]},
        {'label': "5YTD", 'method': "update", 'args': [{"visible":results["5YTD"]} , {'title' : 'Five years normalized stock prices'}]}
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
    
    def run(self):
        """ 
        This method launch directly the app ,
        without having to copy/paste the link from the terminal
        """
        def __open_browser():
            if not os.environ.get("WERKZEUG_RUN_MAIN"):
                webbrowser.open_new('http://127.0.0.1:8050/')

        if __name__ == "__main__":
            Timer(1, __open_browser).start()
            self.app.run_server()


##### Launch the app #####
app = App()
app.run()
print(input('Tap anywhere to close'))

