import datetime as dt 
import base64
import warnings
from typing import Union , Iterable , List
import os
import copy

import pandas as pd , numpy as np
from dash import html 
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader
import yfinance as yf
warnings.filterwarnings('ignore')
yf.pdr_override()



# To not display dash errors
import logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)
from flask import Flask  


class Utils:

    @staticmethod
    def get_data(tickers : Iterable , start_date=dt.datetime(1975,1,1)) -> List[pd.DataFrame]:
        ''' 
        Get the data from yahoo finance , using the list of tickers
        '''
        return [pandas_datareader.data.get_data_yahoo(tick , start_date) for tick in tickers]


    @staticmethod
    def _create_benchmarks_sectors(sector_compositions:dict) -> List[pd.DataFrame]:
        """ 
        Create the list of df with aggregated benchmarks by sector
        """
        sector_df_list = []
        for sector in sector_compositions:

            tickers = sector_compositions[sector]

            sector_df_list.append(pd.DataFrame(pd.concat\
                                                (Utils.get_data(tickers , start_date=dt.datetime.now() - dt.timedelta(days=365*5)),\
                                                 axis=1)["Adj Close"].mean(axis=1) , columns=["Adj Close"]).sort_index())
            
        return sector_df_list 

    @staticmethod
    def get_pct_change(df : pd.DataFrame) -> dict:
        """ 
        Compute the difference in percentage for several 
        period of time , returns a dict with the time period as key
        and percentage change as value 
        """
        current = float(df.iloc[-1]['Adj Close'])


        # Week
        previous , week_diff = Utils.prev_diff(7 , current , df) \
                            or Utils.prev_diff(5, current , df) \
                            or Utils.prev_diff(6, current , df) \
                            or Utils.prev_diff(9, current , df) \
                            or ("N/A" , "N/A")

        # Two weeks
        previous , twoweek_diff = Utils.prev_diff(14, current , df) \
                                or Utils.prev_diff(10, current , df) \
                                or Utils.prev_diff(16, current , df) \
                                or Utils.prev_diff(12, current , df) \
                                or ("N/A" , "N/A")       

        # Six months
        previous , sixmonth_diff = Utils.prev_diff(180, current , df) \
                                or Utils.prev_diff(6*19, current , df) \
                                or Utils.prev_diff(6*20, current , df) \
                                or Utils.prev_diff(6*21, current , df) \
                                or Utils.prev_diff(6*22, current , df) \
                                or ("N/A" , "N/A")

        # One year
        previous , oneyear_diff = Utils.prev_diff(252, current , df) \
                                or Utils.prev_diff(251, current , df) \
                                or Utils.prev_diff(250, current , df) \
                                or Utils.prev_diff(249, current , df) \
                                or Utils.prev_diff(253, current , df) \
                                or ("N/A" , "N/A")


        # Five years
        previous , fiveyear_diff = Utils.prev_diff(365*5, current , df) \
                                or Utils.prev_diff(252*5, current , df) \
                                or Utils.prev_diff(251*5, current , df) \
                                or Utils.prev_diff(250*5, current , df) \
                                or Utils.prev_diff(249*5, current , df) \
                                or Utils.prev_diff(253*5, current , df) \
                                or ("N/A" , "N/A")



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


    @staticmethod
    def minmax_scale(days : int , listestock : Iterable) -> list[pd.DataFrame]:
        ''' Scale the data , to make comparaison possible between stocks '''

        def __fix_date_error(df_ , days):

            day = copy.deepcopy(days)
            counter = 0
            while (counter := counter + 1) < 100_000_000:
                try:
                    df_ = df[df.index[-1] - dt.timedelta(day):]
                    return df_
                except KeyError:
                    day -= 1

        period_df = []
        for df in listestock:

            try:
                df_ = df[df.index[-1] - dt.timedelta(days):]
            except KeyError:
                df_ = __fix_date_error(df , days)


            df_['normalized'] = MinMaxScaler().fit_transform(df_['Adj Close'].values.reshape(-1,1))
            period_df.append(df_)
            
        return period_df



    @staticmethod
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

        return string

    @staticmethod
    def prev_diff(n  , current , df) -> Union[tuple , None]:
        '''Return previous and diff , or None in case of error'''
        try:
            previous = float(df[df.index == str(df.index[-1] - dt.timedelta(n))]['Adj Close'])
            diff = np.round(100*((current - previous) / previous),2)
            return previous , diff
        except:
            return None



    @staticmethod
    def load_image_as_bytes(image_path:str) -> bytes:
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