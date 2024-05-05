from typing import Dict
import datetime
import copy

import pandas as pd 


class DfSubsetter:

    """ 
    The purpose of this class is to preprocess data for the viz (susbet the dataframes and prepare the 'legend_only' dict)
    """

    TIME_PERIOD_LABELS_MAIN = ("ALL" , "2WTD", "1MTD", "3MTD", "6MTD", "1YTD", "5YTD")
    MINUS_TIME_PERIOD = (30*3, 30*5, 30*7, 365, 365*5)
    TIME_PERIOD_LABELS = ("2MTD", "4MTD", "6MTD", "1YTD", "5YTD")

    
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
    def get_legend_argument_all_figs():

        n_time_period = len(DfSubsetter.TIME_PERIOD_LABELS)

        results = {}

        pointer = 0

        base_list_visible = [False for i in range(n_time_period)]
        while pointer < n_time_period:

            base_list_visible_label = copy.deepcopy(base_list_visible)
            base_list_visible_label[pointer] = True
            label = DfSubsetter.TIME_PERIOD_LABELS[pointer]
            results[label] = base_list_visible_label

            pointer += 1 

        return results


    @staticmethod
    def get_legend_arguments_main(len_tickers:int) -> dict:
        
        n_time_period = len(DfSubsetter.TIME_PERIOD_LABELS_MAIN)

        first_n_false = 0
        last_n_false = n_time_period-1

        legend_only_templ = ['legendonly' for i in range(len_tickers - 1)]

        results = {
            
                "ALL" : [ 
                    *['legendonly' for i in range(len_tickers - 1)],
                    *[False for i in range(len_tickers*5)],
                    True,
                ] , 
                

                }

        for label in DfSubsetter.TIME_PERIOD_LABELS_MAIN:

            
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

                {'label': "ALL", 'method': "update", 'args': [{"visible": results["ALL"]} , {'title' : f'Overall normalized stock prices'}]},
                {'label': "2WTD", 'method': "update", 'args': [{"visible": results["2WTD"]} , {'title' : f'Two weeks normalized stock prices'}]},
                {'label': "1MTD", 'method': "update", 'args': [{"visible": results["1MTD"]} , {'title' : f'One month normalized stock prices'}]},
                {'label': "3MTD", 'method': "update", 'args': [{"visible": results["3MTD"]} , {'title' : f'Three months normalized stock prices'}]},
                {'label': "6MTD", 'method': "update", 'args': [{"visible": results["6MTD"]} , {'title' : f'Six months normalized stock prices'}]},
                {'label': "1YTD", 'method': "update", 'args': [{"visible": results["1YTD"]} , {'title' : f'One year normalized stock prices'}]},
                {'label': "5YTD", 'method': "update", 'args': [{"visible":results["5YTD"]} , {'title' : f'Five years normalized stock prices'}]}
                ]
    

