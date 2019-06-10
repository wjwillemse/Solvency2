import operator
import pandas as pd
import numpy as np
import itertools
from functools import reduce

# equivalence -> reported together
def logical_equivalence(*c):
    nonzero_c1 = (c[0] != 0)
    nonzero_c2 = (c[1] != 0)
    return ((nonzero_c1 & nonzero_c2) | (~nonzero_c1 & ~nonzero_c2))

# implication
def logical_implication(*c):
    nonzero_c1 = (c[0] != 0)
    nonzero_c2 = (c[1] != 0)
    return ~(nonzero_c1 & ~nonzero_c2)

def logical_or(*c):
    nonzero_c1 = (c[0] != 0)
    nonzero_c2 = (c[1] != 0)
    return (nonzero_c1 | nonzero_c2)

def logical_and(*c):
    nonzero_c1 = (c[0] != 0)
    nonzero_c2 = (c[1] != 0)
    return (nonzero_c1 & nonzero_c2)

operators = {'>' : operator.gt,
             '<' : operator.lt,
             '>=': operator.ge,
             '<=': operator.le,
             '=' : operator.eq,
             '!=': operator.ne,
             '<->': logical_equivalence,
             '-->': logical_implication}

preprocess = {'>':   operator.and_,
              '<':   operator.and_,
              '>=':  operator.and_,
              '<=':  operator.and_,
              '=' :  operator.and_,
              '!=':  operator.and_,
              '<->': operator.or_,
              '-->': operator.or_,
              'sum': operator.and_}

def prepare_dataframe(df):
    # we only look at numerical columns
    df = df[[df.columns[c] for c in range(len(df.columns)) 
             if (df.dtypes[c] == 'float64' or df.dtypes[c] == 'int64')]]    
    # remove columns with only zeros
    df = df.loc[:, (df != 0).any(axis=0)]
    return df

def derive_pattern_statistics(co):
    # co_sum is the support of the pattern
    co_sum = co.sum()
    ex_sum = (~co).sum()
    # conf is the confidence of the pattern
    conf = np.round(co_sum / (co_sum + ex_sum), 4)
    # oddsratio is a correlation measure
    #oddsratio = (1 + co_sum) / (1 + ex_sum)
    return co_sum, ex_sum, conf #, oddsratio

def derive_pattern_data(df, P_columns, Q_columns, co, confidence, include_co_ex, data_filter):
    data = list()
    # pattern statistics
    co_sum, ex_sum, conf = derive_pattern_statistics(co)
    # we only store the rules with confidence higher than conf
    if conf >= confidence:
        data = [P_columns, Q_columns, co_sum, ex_sum, conf]
        if include_co_ex:
            if data_filter is None:
                data.extend([list(df.index[co]), list(df.index[~co])])
            else:
                data.extend([list(df.index[data_filter][co]), list(df.index[data_filter][~co])])
    return data

# generate patterns of the form [c1] operator value where c1 in df.columns
# operators:
# '!=' with 0 -> patterns in reported columns
# '=' with 0 -> patterns in not reported columns
# '>' with 0 -> patterns in reported columns with strictly positive values
# '<' with 0 -> patterns in reported columns with strictly negative values

def patterns_column_value(dataframe = None, 
                          pattern   = None,
                          P_columns = None,
                          value     = None,
                          confidence= 0.75, 
                          include_co_ex = False):
    
    dataframe = prepare_dataframe(dataframe)
    data_array = dataframe.values.T
    
    if P_columns is None: 
        P_columns = range(len(dataframe.columns))
    else:
        P_columns = [dataframe.columns.get_loc(i) for i in P_columns]
    
    for c in P_columns:
        # confirmations and exceptions of the pattern, a list of booleans
        co = reduce(operators[pattern], [data_array[c, :], 0])
        pattern_data = derive_pattern_data(dataframe,
                                [dataframe.columns[c]],
                                value, 
                                co, 
                                confidence,
                                include_co_ex, None)
        if pattern_data:
            yield [pattern] + pattern_data
            
# generate patterns of the form {[c1] operator [c2]} where c1 and c2 in df.columns
# operators:
# '=' -> patterns in equal values in columns
# '<' -> patterns in lower values in columns
# '>' -> patterns in greater values in columns
# '<->' -> patterns in datapoints that are reported together

def patterns_column_column(dataframe  = None,
                           pattern    = None,
                           P_columns  = None, 
                           Q_columns  = None, 
                           confidence = 0.75, 
                           include_co_ex = False):
    
    preprocess_operator = preprocess[pattern]
    
    dataframe = prepare_dataframe(dataframe)
    # set up boolean masks for nonzero items per column
    nonzero = (dataframe.values != 0).T
    
    # if no columns are given, all columns are checked
    if P_columns is None: 
        P_columns = range(len(dataframe.columns))
    else:
        P_columns = [dataframe.columns.get_loc(i) for i in P_columns]
    
    if Q_columns is None: 
        Q_columns = range(len(dataframe.columns))
    else:
        Q_columns = [dataframe.columns.get_loc(i) for i in Q_columns]
    
    for c0 in P_columns:
        for c1 in Q_columns:
            if c0 != c1:
                # applying the filter
                data_filter = reduce(preprocess_operator, [nonzero[c] for c in [c0, c1]])
                data_array = dataframe.values[data_filter].T
                if data_array.size:
                    # confirmations of the pattern, a list of booleans
                    co = reduce(operators[pattern], data_array[[c0, c1], :])
                    pattern_data = derive_pattern_data(dataframe,
                                        [dataframe.columns[c0]], 
                                        [dataframe.columns[c1]], 
                                        co, 
                                        confidence,
                                        include_co_ex, data_filter)
                    if pattern_data:
                        yield [pattern] + pattern_data

def patterns_sums_column(dataframe = None,
                         pattern = None,
                         confidence = 0.75, 
                         sum_elements = 2, 
                         include_co_ex = False):

    preprocess_operator = preprocess[pattern]

    df = prepare_dataframe(dataframe)
    data_array = df.values.T

    # set up boolean masks for nonzero items per column
    nonzero = (df.values != 0).T

    n = len(df.columns)
    matrix = np.zeros(shape = (n, n), dtype = bool)
    for c in itertools.combinations(range(n), 2):
        v = (data_array[c[1], :] <= data_array[c[0], :] + 1).all()
        matrix[c[0], c[1]] = v
        matrix[c[1], c[0]] = ~v
    np.fill_diagonal(matrix, False)

    for elements in range(2, sum_elements + 1):
        for sum_col in range(n):
            lower_columns, = np.where(matrix[sum_col] == True)
            for sum_parts in itertools.combinations(lower_columns, elements):
                
                subset = sum_parts + (sum_col,)

                data_filter = reduce(preprocess_operator, [nonzero[c] for c in subset])
                data_array = df.values[data_filter].T
                
                if data_array.size:
                    # determine sum of columns in subset
                    data_array[sum_col, :] = -data_array[sum_col, :]
                    co = (abs(reduce(operator.add, data_array[subset, :])) < 1)
                    pattern_data = derive_pattern_data(df, 
                                        [df.columns[c] for c in sum_parts],
                                        [df.columns[sum_col]],
                                        co, 
                                        confidence,
                                        include_co_ex, None)
                    if pattern_data:
                        yield [pattern] + pattern_data

def generate(dataframe  = None,
             pattern    = None,
             P_columns  = None, 
             Q_columns  = None,
             value      = None,
             sum_elements = None,
             confidence = 0.75, 
             include_co_ex = False):

    # if a value is given -> columns pattern value
    if not value is None:
        return patterns_column_value(dataframe = dataframe,
                                     pattern = pattern,
                                     P_columns = P_columns,
                                     value = value,
                                     confidence = confidence,
                                     include_co_ex = include_co_ex)
    # if the pattern is sum and sum_elements is given -> c1 + ... cn = c
    elif pattern == 'sum' and not sum_elements is None:
        return patterns_sums_column(dataframe = dataframe,
                                    pattern = pattern,
                                    confidence = confidence, 
                                    sum_elements = sum_elements)
    # everything else -> c1 pattern c2
    else:
        return patterns_column_column(dataframe = dataframe,
                                     pattern = pattern, 
                                     P_columns = P_columns,
                                     Q_columns = Q_columns,
                                     confidence = confidence,
                                     include_co_ex = include_co_ex)