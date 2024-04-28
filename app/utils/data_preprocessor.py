from typing import Dict
import datetime

import pandas as pd 


class DfSubsetter:

    """ 
    The purpose of this class is to preprocess data for the viz (susbet the dataframes and prepare the 'legend_only' dict)
    """

    MINUS_TIME_PERIOD = (14, 30, 30*6, 365, 365*5)
    TIME_PERIOD_LABELS = ("2WTD", "1MTD", "6MTD", "1YTD", "5YTD")

    def __init__(self, df:pd.DataFrame , 
                title:str , 
                len_ticker:int):
        
        self.df = df 
        self.title = title 
        self.len_ticker = len_ticker


    def main(self):

        subset_dict:dict = DfSubsetter.get_df_subsets_dict(df=self.df)
        legend_argument_dict:dict = DfSubsetter.get_legend_arguments(title=self.title, len_tickers=self.len_ticker)

        return subset_dict, legend_argument_dict 
    
    @staticmethod
    def get_df_subsets_dict(df:pd.DataFrame) -> Dict[str, pd.DataFrame]:

        latest_time_period:pd.Timestamp = df.index[-1]

        results = {}

        minus_t:int
        label:str
        for minus_t, label in zip(DfSubsetter.MINUS_TIME_PERIOD, 
                                  DfSubsetter.TIME_PERIOD_LABELS):

            start_date = latest_time_period - datetime.timedelta(minus_t)
            subset_df = df[start_date:]
            results[label] = subset_df

        return results    
    
    @staticmethod
    def get_legend_arguments(title:str , len_tickers:int) -> dict:
        
        n_time_period = len(DfSubsetter.TIME_PERIOD_LABELS)

        first_n_false = 1
        last_n_false = n_time_period-1

        legend_only_templ = ['legendonly' for i in range(len_tickers - 1)]

        results = {}

        for label in DfSubsetter.TIME_PERIOD_LABELS:

            
            result = [ 

                *[False for i in range(len_tickers * first_n_false)] , 
                *legend_only_templ, 
                True,
                *[False for i in range(len_tickers * last_n_false)]
            ]
            

            first_n_false += 1
            last_n_false -= 1

        
            results[label] = result


        # Create the buttons
        return  [
                {'label': "2WTD", 'method': "update", 'args': [{"visible": results["2WTD"]} , {'title' : f'Two weeks {title}'}]},
                {'label': "1MTD", 'method': "update", 'args': [{"visible": results["1MTD"]} , {'title' : f'One month {title}'}]},
                {'label': "6MTD", 'method': "update", 'args': [{"visible": results["6MTD"]} , {'title' : f'Six months {title}'}]},
                {'label': "1YTD", 'method': "update", 'args': [{"visible": results["1YTD"]} , {'title' : f'One year {title}'}]},
                {'label': "5YTD", 'method': "update", 'args': [{"visible":results["5YTD"]} , {'title' : f'Five years {title}'}]}
                ]

    