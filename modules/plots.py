import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def mean_data_aggregation(data: pd.core.frame.DataFrame, response: str, predictor: list, unknown: list, fitted: str = None) -> tuple:
    """
    Aggregate means accross groups

    :param data: dataframe from which we could use 2 or more columns - response variable and specified predictors. Predictors must be categoric 
        for this type of graph.
    :param response: string with the name of response column.
    :param predictor: string with the name of predictor that we are intersted in to plot.
    :param unknown: value to be used as replacement of null values in predictors
    :param fitted: name of column with model prediction, if None no model prediction will be displayed
    :return: tuple with groups, counts of observations and means.
    """
    

    for index, col in enumerate(predictor):

        if data[col].dtype == "category":
            data[col] = data[col].astype(str)
            
        if len(unknown) == len(predictor):
            data[col].fillna(unknown[index], inplace = True)
        else:
            data[col].fillna(unknown[0], inplace = True)

    grp = data.groupby(predictor, dropna=False)
    # grp_fit = data.groupby(fitted, dropna=False)
    grp_levels = grp[predictor].groups.keys()
    if len(predictor)>1:
        grp_levels = ["_".join([str(group)]) for group in grp_levels]
    
    # replace NA for 0 in potentialy empty categories
    means = grp[response].mean()
    means = [0 if pd.isna(x) else x for x in means.values]
    
    if fitted:
        fitted_means = grp[fitted].mean()
        fitted_means = [0 if pd.isna(x) else x for x in fitted_means.values]
    else:
        fitted_means = None

    cnts = grp[response].count()

    return list(grp_levels), means, fitted_means, list(cnts.values)


def mean_count_plot(data: pd.core.frame.DataFrame, response: str, predictor: list, unknown: list, fitted: str = None) -> go.Figure:
    """
    Plot data as graph with bars representing counts of value for each category and points as mean of response for this category.

    :param data: dataframe from which we will use only 2 columns - response variable and predictor. Predictor must be categoric for this
        type of graph.
    :param response: string with the name of response column.
    :param predictor: string with the name of predictor that we are intersted in to plot.
    :param unknown: value to be used as replacement of null values in predictors
    :param fitted: name of column with model prediction, if None no model prediction will be displayed
    :return: plotly figure.
    """
    grp_levels, grp_means, grp_fit, grp_cnts = mean_data_aggregation(data, response, predictor, unknown, fitted)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Bar(x=grp_levels, y=grp_cnts, yaxis='y2', name='Data Counts', marker=dict(color='yellow')),
        secondary_y=False,
    )
    
    if fitted:
        fig.add_trace(
        go.Scatter(x=grp_levels, y=grp_means, mode='markers', name='Mean Values', marker=dict(color='red', size=30)),
        secondary_y=True,
        )
        
        fig.add_trace(
        go.Scatter(x=grp_levels, y=grp_fit, mode='markers', name='Mean Fitted', marker=dict(color='green', size=30, symbol='triangle-up')),
        secondary_y=True,
        )
    else:
        fig.add_trace(
        go.Scatter(x=grp_levels, y=grp_means, mode='markers', name='Mean Values', marker=dict(color='red', size=30)),
        secondary_y=True,
        )
    predictor_name = predictor[0]
    if len(predictor)>1:
        predictor_name = '_'.join(predictor)

    # Add figure title
    fig.update_layout(
        title_text=f"Data aggregated by {predictor_name}"
    )

    # Set x-axis title
    fig.update_xaxes(title_text=predictor_name)

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Data Counts</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Data Means</b>", secondary_y=True)

    return fig


def investigate_categoric_variable(data: pd.core.frame.DataFrame, response: str, predictor: list, 
                                   unknown: list, verbose: bool = True, fitted: str = None):
    """
    Do basic variable analysis.

    :param data: dataframe from which we will use only 2 columns - response variable and predictor. Predictor must be categoric for this
        type of graph.
    :param response: string with the name of response column.
    :param predictor: string with the name of predictor that we are intersted in to plot.
    :param unknown: value to be used as replacement of null values in predictors
    :param verbose: bool, if True plot will be displayed and information printed
    :param fitted: name of column with model prediction, if None no model prediction will be displayed
    :return: returns nothing
    """
    grp_levels, grp_means, grp_fit, grp_cnts = mean_data_aggregation(data, response, predictor, unknown, fitted)

    # To allow better comprehension of this data we can create plot
    fig = mean_count_plot(data, response, predictor, unknown, fitted)
    
    if verbose:
        print(f"This variable contains these levels: {grp_levels}")
        print(f"Number of observation by level of this variable: {grp_cnts}")
        print(f"Mean value of response by level of this variable: {grp_means}")
        print(f"Mean fitted value by level of this variable: {grp_fit}")
        # fig.show()
    
    return fig
    