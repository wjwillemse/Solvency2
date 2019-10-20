import operator
import pandas as pd
from pandas import DataFrame
import numpy as np
import itertools
from fractions import Fraction
from functools import reduce

class PatternGenerator(DataFrame):
    
    _metadata = ['pattern_definitions', 'pattern_results']

    def __init__(self, *args, **kwargs):
        self.pattern_definitions = None
        self.pattern_results = None
        super(PatternGenerator, self).__init__(*args, **kwargs)

    def __setattr__(self, attr, val):
        # have to special case b/c pandas tries to use as column
        if attr == 'pattern_definitions':
            object.__setattr__(self, attr, val)
        elif attr == 'pattern_results':
            object.__setattr__(self, attr, val)
        else:
            super(PatternGenerator, self).__setattr__(attr, val)

    def generate_patterns(self, *args, **kwargs):
        append = kwargs.pop('append', False)
        if append == True and self.pattern_definition is not None:
            # from struct or from pattern
            self.pattern_definitions.extend(list(generate(dataframe = self, *args, **kwargs)))
            self.pattern_definitions = list(set(self.pattern_definitions))
        else:
            self.pattern_definitions = list(generate(dataframe = self, *args, **kwargs))

# equivalence -> reported together
def logical_equivalence(*c):
    """Operator definition of logical equivalence taking two parameters
    """
    nonzero_c1 = (c[0] != 0)
    nonzero_c2 = (c[1] != 0)
    return ((nonzero_c1 & nonzero_c2) | (~nonzero_c1 & ~nonzero_c2))

# implication
def logical_implication(*c):
    """Operator definition of logical implication taking two parameters
    """
    nonzero_c1 = (c[0] != 0)
    nonzero_c2 = (c[1] != 0)
    return ~(nonzero_c1 & ~nonzero_c2)

def logical_or(*c):
    """Operator definition of logical or taking two parameters
    """
    nonzero_c1 = (c[0] != 0)
    nonzero_c2 = (c[1] != 0)
    return (nonzero_c1 | nonzero_c2)

def logical_and(*c):
    """Operator definition of logical and taking two parameters
    """
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
              'sum': operator.and_,
              'ratio': operator.and_,
              'interval': operator.and_}

def derive_pattern_statistics(co):
    """Pattern statistics:
       co_sum: support (number of confirmations)
       ex_sum: number of exceptions
       conf  : confidence
    """
    co_sum = co.sum()
    ex_sum = (~co).sum()
    conf = np.round(co_sum / (co_sum + ex_sum), 4)
    # oddsratio is a correlation measure
    #oddsratio = (1 + co_sum) / (1 + ex_sum)
    return co_sum, ex_sum, conf #, oddsratio

def derive_pattern_data(dataframe, 
                        P, 
                        Q,
                        pattern, 
                        co, 
                        confidence, 
                        include_co,
                        include_ex, 
                        data_filter):
    """Derives the pattern data of a single pattern
       Output: [[pattern, P, Q], co_sum, ex_sum, conf (,confirmation)(,exceptions)]
    """

    data = list()
    # pattern statistics
    co_sum, ex_sum, conf = derive_pattern_statistics(co)
    # we only store the rules with confidence higher than conf
    if conf >= confidence:
        data = [[pattern, P, Q], co_sum, ex_sum, conf]
        if include_co:
            if data_filter is None:
                data.extend([list(dataframe.index[co])])
            else:
                data.extend([list(dataframe.index[data_filter][co])])
        if include_ex:
            if data_filter is None:
                data.extend([list(dataframe.index[~co])])
            else:
                data.extend([list(dataframe.index[data_filter][~co])])
    return data

def get_parameters(parameters):
    """Extract parameters from parameters list
    """

    confidence = parameters.get("min_confidence", 0.75)
    support    = parameters.get("min_support", 1)
    include_co = parameters.get("include_co", False)
    include_ex = parameters.get("include_ex", False)

    return confidence, support, include_co, include_ex


def patterns_column_value(dataframe = None, 
                          pattern   = None,
                          columns   = None,
                          value     = None,
                          parameters= {}):
    """Generate patterns of the form "[c1] operator value" where c1 is in columns
    """

    confidence, support, include_co, include_ex = get_parameters(parameters)

    data_array = dataframe.values.T
    
    for c in columns:
        # confirmations and exceptions of the pattern, a list of booleans
        co = reduce(operators[pattern], [data_array[c, :], 0])
        pattern_data = derive_pattern_data(dataframe,
                                           dataframe.columns[c],
                                           value,
                                           pattern, 
                                           co, 
                                           confidence,
                                           include_co, 
                                           include_ex, None)
        if pattern_data and len(co) >= support:
            yield pattern_data
            

def patterns_column_column(dataframe  = None,
                           pattern    = None,
                           P_columns  = None, 
                           Q_columns  = None, 
                           parameters = {}):
    """ Generate patterns of the form {[c1] operator [c2]} where c1 and c2 in df.columns
        operators:
        '=' -> patterns in equal values in columns
        '<' -> patterns in lower values in columns
        '>' -> patterns in greater values in columns
        '<->' -> patterns in datapoints that are reported together
    """
    
    confidence, support, include_co, include_ex = get_parameters(parameters)

    preprocess_operator = preprocess[pattern]
    
    initial_data_array = dataframe.values.T
    # set up boolean masks for nonzero items per column
    nonzero = (initial_data_array != 0)
    
    for c0 in P_columns:
        for c1 in Q_columns:
            if c0 != c1:
                # applying the filter
                data_filter = reduce(preprocess_operator, [nonzero[c] for c in [c0, c1]])
                if data_filter.any():
                    data_array = initial_data_array[:, data_filter]
                    # confirmations of the pattern, a list of booleans
                    co = reduce(operators[pattern], data_array[[c0, c1], :])
                    if co.any():
                        pattern_data = derive_pattern_data(dataframe,
                                            dataframe.columns[c0], 
                                            dataframe.columns[c1], 
                                            pattern,
                                            co, 
                                            confidence,
                                            include_co,
                                            include_ex, data_filter)
                        if pattern_data and len(co) >= support:
                            yield pattern_data

def patterns_ratio(dataframe  = None,
                   pattern    = None,
                   columns    = None, 
                   parameters = {}):
    """Generate patterns with ratios
    """
    
    confidence, support, include_co, include_ex = get_parameters(parameters)

    limit_denominator = parameters.get("limit_denominator", 10000000)

    preprocess_operator = preprocess[pattern]
    
    # set up boolean masks for nonzero items per column
    nonzero = (dataframe.values != 0).T
    
    for c0 in columns:
        for c1 in columns:
            if c0 != c1:
                # applying the filter
                data_filter = reduce(preprocess_operator, [nonzero[c] for c in [c0, c1]])
                data_array = map(lambda e: Fraction(e).limit_denominator(limit_denominator), 
                                 dataframe.values[data_filter, c0] / dataframe.values[data_filter, c1])
                ratios = pd.Series(data_array)
                if support >= 2:
                    possible_ratios = ratios.loc[ratios.duplicated(keep = False)].unique()
                else:
                    possible_ratios = ratios.unique()
                for v in possible_ratios:
                    if (abs(v) > 1e-6) and (v > -1) and (v < 1):
                        # confirmations of the pattern, a list of booleans
                        co = ratios==v
                        if sum(co) >= support:
                            pattern_data = derive_pattern_data(dataframe,
                                            str(v),
                                            [dataframe.columns[c0], 
                                            dataframe.columns[c1]], 
                                            pattern,
                                            co, 
                                            confidence,
                                            include_co,
                                            include_ex, data_filter)
                            if pattern_data:
                                yield pattern_data

def patterns_interval(dataframe  = None,
                      pattern    = None,
                      columns    = None, 
                      parameters = {}):
    
    confidence, support, include_co, include_ex = get_parameters(parameters)

    limit_denominator = parameters.get("limit_denominator", 10000000)

    preprocess_operator = preprocess[pattern]
    
    # set up boolean masks for nonzero items per column
    nonzero = (dataframe.values != 0).T
    
    for c0 in columns:
        for c1 in columns:
            if c0 != c1:
                # applying the filter
                data_filter = reduce(preprocess_operator, [nonzero[c] for c in [c0, c1]])
                data_array = map(lambda e: Fraction(e).limit_denominator(limit_denominator), 
                                 dataframe.values[data_filter, c0] / dataframe.values[data_filter, c1])
                ratios = pd.Series(data_array)
                
                interval_elements = list(ratios.loc[ratios.duplicated(keep = False)].unique())

                if 1 in interval_elements: interval_elements.remove(1)
                if -1 in interval_elements: interval_elements.remove(-1)

                if len(interval_elements) > 0:
                    interval_elements.sort()
                    if min(ratios) not in interval_elements:
                        interval_elements = [-np.inf] + interval_elements
                    if max(ratios) not in interval_elements:
                        interval_elements = interval_elements + [np.inf]
                    intervals = list(itertools.combinations(interval_elements,2))
                    if (-np.inf, np.inf) in intervals:
                        intervals.remove((-np.inf, np.inf))
                
                    for interval in intervals:
                        # confirmations of the pattern, a list of booleans
                        co = (ratios >= interval[0]) & (ratios <= interval[1])
                        if sum(co) >= support:
                            pattern_data = derive_pattern_data(dataframe,
                                               "[" + str(interval[0]) + ", " + str(interval[1]) + "]",
                                                [dataframe.columns[c0], 
                                                dataframe.columns[c1]], 
                                                pattern,
                                                co, 
                                                confidence,
                                                include_co,
                                                include_ex, data_filter)
                            if pattern_data:
                                yield pattern_data

def patterns_sums_column(dataframe  = None,
                         pattern    = None,
                         parameters = {}):
    """Generate patterns with sums
    """

    confidence, support, include_co, include_ex = get_parameters(parameters)
    sum_elements = parameters.get("sum_elements", 2)

    preprocess_operator = preprocess[pattern]
    initial_data_array = dataframe.values.T
    # set up boolean masks for nonzero items per column
    nonzero = (initial_data_array != 0)

    n = len(dataframe.columns)
#    matrix = np.ones(shape = (n, n), dtype = bool)
#    for c in itertools.combinations(range(n), 2):
#        v = (data_array[c[1], :] <= data_array[c[0], :] + 1).all()
#        matrix[c[0], c[1]] = v
#        matrix[c[1], c[0]] = ~v
#    np.fill_diagonal(matrix, False)

    for lhs_elements in range(2, sum_elements + 1):
        for rhs_column in range(n):
            start_array = initial_data_array
            # minus righthandside is taken so we can use sum/add function for all columns
            start_array[rhs_column, :] = -start_array[rhs_column, :]
#            lower_columns, = np.where(matrix[sum_col] == True)
            lhs_column_list = [col for col in range(n) if col != rhs_column]
            for lhs_columns in itertools.combinations(lhs_column_list, lhs_elements):
                all_columns = lhs_columns + (rhs_column,)
                #data_filter = reduce(preprocess_operator, [nonzero[c] for c in all_columns])
                data_filter = np.logical_and.reduce(nonzero[all_columns, :])
#                data_filter = nonzero[rhs_column]
#                for col in lhs_columns:
#                    data_filter = data_filter & nonzero[col]
                if data_filter.any():
                    data_array = start_array[:, data_filter]
#                    co = (abs(reduce(operator.add, data_array[all_columns, :])) < 1)
                    co = (abs(np.sum(data_array[all_columns, :], axis = 0)) < 1)
                    if co.any():
                        pattern_data = derive_pattern_data(dataframe, 
                                            [dataframe.columns[c] for c in lhs_columns],
                                            dataframe.columns[rhs_column],
                                            pattern,
                                            co, 
                                            confidence,
                                            include_co,
                                            include_ex, None)
                        if pattern_data and len(co) >= support:
                            yield pattern_data

def generate(dataframe   = None,
             P_dataframe = None,
             Q_dataframe = None,
             pattern     = None,
             columns     = None,
             P_columns   = None, 
             Q_columns   = None,
             value       = None,
             parameters  = {}):
    """General function to call specific pattern functions
       Only numerical columns are used
    """

    # if P_dataframe and Q_dataframe are given then join the dataframes and select columns
    if (not P_dataframe is None) and (not Q_dataframe is None):
        try:
            dataframe = P_dataframe.join(Q_dataframe)
        except:
            print("Join of P_dataframe and Q_dataframe failed, overlapping columns?")
            return []
        P_columns = P_dataframe.columns
        Q_columns = Q_dataframe.columns

    # select all columns with numerical values
    numerical_columns = [dataframe.columns[c] for c in range(len(dataframe.columns)) 
                            if ((dataframe.dtypes[c] == 'float64') or (dataframe.dtypes[c] == 'int64')) and (dataframe.iloc[:, c] != 0).any()]
    dataframe = dataframe[numerical_columns]

    if not P_columns is None:
        P_columns = [dataframe.columns.get_loc(c) for c in P_columns if c in numerical_columns]
    else:
        P_columns = range(len(dataframe.columns))

    if not Q_columns is None:
        Q_columns = [dataframe.columns.get_loc(c) for c in Q_columns if c in numerical_columns]
    else:
        Q_columns = range(len(dataframe.columns))

    if not columns is None: 
        columns = [dataframe.columns.get_loc(c) for c in columns if c in numerical_columns]
    else:
        columns = range(len(dataframe.columns))

    # if a value is given -> columns pattern value
    if not value is None:
        return patterns_column_value(dataframe = dataframe,
                                     pattern = pattern,
                                     columns = columns,
                                     value = value,
                                     parameters = parameters)
    # if the pattern is sum and sum_elements is given -> c1 + ... cn = c
    elif pattern == 'sum':
        return patterns_sums_column(dataframe = dataframe,
                                    pattern = pattern,
                                    parameters = parameters) 
    elif pattern == 'ratio':
        return patterns_ratio(dataframe = dataframe,
                              pattern = pattern,
                              columns = columns,
                              parameters = parameters) 
    elif pattern == 'interval':
        return patterns_interval(dataframe = dataframe,
                                 pattern = pattern,
                                 columns = columns,
                                 parameters = parameters) 
    # everything else -> c1 pattern c2
    else:
        return patterns_column_column(dataframe = dataframe,
                                      pattern = pattern, 
                                      P_columns = P_columns,
                                      Q_columns = Q_columns,
                                      parameters = parameters)