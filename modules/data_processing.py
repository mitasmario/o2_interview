from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import numpy as np

def replace_na(data: pd.core.frame.DataFrame, cols: list, unknown: list) -> pd.core.frame.DataFrame:
    """
    Replace NA values with given value. In case of numerics, replace with zero and create dummy variable for unknown.

    :param data: dataframe
    :param cols: list with column names which will be transformed
    :param unknown: list with values to be used as NA replacements
    :return: dataframe with NAs replaced by value from uknown list
    """
    data_result = data
    for index, col in enumerate(cols):
        if data_result[col].dtype in ["int", "float"]:
            data_result[col + "_unknown"] = np.where(data[col].isnull(), 1, 0)
            data_result[col].fillna(0, inplace = True)
        else:
            data_result[col].fillna(unknown[index], inplace = True)

    return data_result


def encode_categoric_variables(data: pd.core.frame.DataFrame, categoric: list, ordered: list) -> pd.core.frame.DataFrame:
    """
    Transform categoric variables into dummy variable each level encoded by values 1 and 0. Transform ordered variables into int
    type where each level will be numerically ordered as in alphabetical order so be carefull about this fact!

    :param data: dataframe 
    :param categoric: list of categoric variables, these will be turned into dummy variables
    :param ordered: list of ordered variables, these will be turned into ordered numerical variable
    :return: dataframe with old as well as new encoded fields
    """
    # backup original dataframe
    data_transformed = data

    # check whether some of categoric variables does have just 2 distinct levels
    ordered_adj = ordered
    categoric_adj = []
    for var in categoric:
        if data[var].nunique()<=2:
            ordered_adj.append(var)
        else:
            categoric_adj.append(var)

    if ordered_adj:
        le = LabelEncoder()
        data_encoded = data[ordered_adj]
        for column in data_encoded:
            le.fit(data_encoded[column])
            data_transformed[column + "_encoded"] = le.transform(data_encoded[column].astype(str))
    
    if categoric_adj:
        lb = LabelBinarizer()
        data_encoded = data[categoric_adj]
        for column in data_encoded:
            label_binarizer_output = lb.fit_transform(data_encoded[column].astype(str))
            # creating a data frame from the object
            dummy_result = pd.DataFrame(label_binarizer_output, columns = [column + "_" + level + "_encoded" for level in lb.classes_])
            data_transformed = pd.concat([data_transformed, dummy_result], axis=1)

    return data_transformed


def bin_numeric_variable(data: pd.core.frame.DataFrame, bins: list, labels: list, variable: str) -> pd.core.frame.DataFrame:
    """
    Create ordered categoric variable from numeric given edge value and level labels.
    
    :param data: dataframe
    :param bins: numeric list with bin edge values
    :param labels: list with labels for variable levels, recommendation is to alphabetically order levels (for example with numbering)
    :param variable: string with variable name
    :return: dataframe with additional binned variable (with suffix '_binned')
    """
    data_result = data
    data_result[variable + "_binned"] = pd.cut(x = data[variable], bins = bins, labels = labels, include_lowest = True)

    return data_result

