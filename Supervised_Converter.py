import pandas as pd
import numpy as np


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))

        names += [('Prev_{0}'.format(dataset.columns[j])) for j in range(n_vars)]

    for i in range(0, n_out):

        cols.append(df.shift(-i))

        if i == 0:
            names += [('Curr_{0}'.format(dataset.columns[j])) for j in range(n_vars)]
        else:
            names += [('%d_t+%d' % (j + 1, i)) for j in range(n_vars)]

    agg = pd.concat(cols, axis=1)
    agg.columns = names

    if dropnan:
        agg.dropna(inplace=True)
    return agg


dataset = pd.read_csv('./timed.csv')

values = dataset.values

reframed = series_to_supervised(values, 1, 1)

remove_var_list = ['Curr_Clearances','Curr_Clangers','Curr_ContendedPossessions','Curr_UncontendedPossessions',	'Curr_ContestedMarks',	'Curr_MarksInside50'	,'Curr_GoalAssists', 'Curr_Team'	,'Curr_Season'	,'Curr_Margin'	,'Curr_Score'	,'Curr_Disposals'	,'Curr_OnePercenters'	,'Curr_Kicks'	,'Curr_Marks'	,'Curr_Handballs'	,'Curr_Hitouts',	'Curr_Tackles',	'Curr_Rebound50s',	'Curr_Inside50s',	'Curr_Goals',	'Curr_cusum_margin',	'Curr_cusum_Score']

reframed.drop(remove_var_list, axis=1, inplace=True)


reframed.to_csv('./Performance_based_Model_2_Data.csv')
