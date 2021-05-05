import os
import pandas as pd
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------

def first_round():
    """
    :return: list with two values
    >>> out = first_round()
    >>> isinstance(out, list)
    True
    >>> out[0] < 1
    True
    >>> out[1] is "NR" or out[1] is "R"
    True
    """
    return [0.108, 'NR']


def second_round():
    """
    :return: list with three values
    >>> out = second_round()
    >>> isinstance(out, list)
    True
    >>> out[0] < 1
    True
    >>> out[1] is "NR" or out[1] is "R"
    True
    >>> out[2] is "ND" or out[2] is "D"
    True
    """
    return [0.02, 'R', 'ND']


# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def verify_child(heights):
    """
    Returns a series of p-values assessing the missingness
    of child-height columns on father height.

    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> heights = pd.read_csv(fp)
    >>> out = verify_child(heights)
    >>> out['child_50'] < out['child_95']
    True
    >>> out['child_5'] > out['child_50']
    True
    """
    n_repetitions = 500
    cols = ['child_95', 'child_90', 'child_75', 'child_50', 'child_25',
            'child_10',
            'child_5']

    heights_copy = heights.copy()
    pvals = []

    for col in cols:
        heights_copy[col + '_is_null'] = heights[col].isnull()

    for col in cols:
        ks_values = []
        obs_ksX = stats.ks_2samp(
            heights_copy.groupby(col + '_is_null')['father'].get_group(True),
            heights_copy.groupby(col + '_is_null')['father'].get_group(False)
        ).statistic

        for _ in range(n_repetitions):
            shuffled_father = (
                heights_copy['father']
                    .sample(replace=False, frac=1)
                    .reset_index(drop=True)
            )
            shuffled = (
                heights_copy
                    .assign(**{'Shuffled_father': shuffled_father})
            )
            ks = stats.ks_2samp(
                shuffled.groupby(col + '_is_null')['Shuffled_father']
                    .get_group(True),
                shuffled.groupby(col + '_is_null')['Shuffled_father']
                    .get_group(False)
            ).statistic

            ks_values.append(ks)
        pval = np.mean(np.array(ks_values) > obs_ksX)
        pvals.append(pval)

    return pd.Series(index=cols, data=pvals)


def missing_data_amounts():
    """
    Returns a list of multiple choice answers.

    :Example:
    >>> set(missing_data_amounts()) <= set(range(1,6))
    True
    """

    return [5]


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

def cond_single_imputation(new_heights):
    """
    cond_single_imputation takes in a dataframe with columns 
    father and child (with missing values in child) and imputes 
    single-valued mean imputation of the child column, 
    conditional on father. Your function should return a Series.

    :Example:
    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> df = pd.read_csv(fp)
    >>> df['child'] = df['child_50']
    >>> out = cond_single_imputation(df)
    >>> out.isnull().sum() == 0
    True
    >>> (df.child.std() - out.std()) > 0.5
    True
    """
    fill = new_heights \
        .groupby(pd.qcut(new_heights['father'], 4)) \
        .transform('mean')

    return new_heights.fillna(fill)['child']


# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------


def quantitative_distribution(child, N):
    """
    quantitative_distribution that takes in a Series and an integer 
    N > 0, and returns an array of N samples from the distribution of 
    values of the Series as described in the question.
    :Example:
    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> df = pd.read_csv(fp)
    >>> child = df['child_50']
    >>> out = quantitative_distribution(child, 100)
    >>> out.min() >= 56
    True
    >>> out.max() <= 79
    True
    >>> np.isclose(out.mean(), child.mean(), atol=1)
    True
    """
    emp_dist = child.dropna()
    hist = np.histogram(emp_dist, bins=10)

    vals = []
    for i in range(N):
        interval = np.random.choice(hist[1][1:], p=hist[0] / hist[0].sum())
        idx_h = np.where(hist[1] == interval)[0][0]
        idx_l = idx_h - 1
        sample = child.loc[
            (child <= hist[1][idx_h]) & (child >= hist[1][idx_l])]
        vals.append(np.random.choice(sample, replace=True))

    return np.array(vals)


def impute_height_quant(child):
    """
    impute_height_quant takes in a Series of child heights 
    with missing values and imputes them using the scheme in
    the question.

    :Example:
    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> df = pd.read_csv(fp)
    >>> child = df['child_50']
    >>> out = impute_height_quant(child)
    >>> out.isnull().sum() == 0
    True
    >>> np.isclose(out.mean(), child.mean(), atol=0.5)
    True
    """
    N = child.isnull().sum()
    fill_values = quantitative_distribution(child, N).reshape(N)
    fill_idx = child.loc[child.isnull()].index

    return child.fillna(pd.Series(data=fill_values, index=fill_idx))


# ---------------------------------------------------------------------
# Question # X
# ---------------------------------------------------------------------

def answers():
    """
    Returns two lists with your answers
    :return: Two lists: one with your answers to multiple choice questions
    and the second list has 6 websites that satisfy given requirements.
    >>> list1, list2 = answers()
    >>> len(list1)
    4
    >>> len(list2)
    6
    """
    return [1, 2, 2, 1], \
           ["https://www.emuparadise.me",
            "https://froglanders.com",
            "https://www.icecreamshoplajolla.com",
            "https://www.netflix.com",
            "https://www.gradescope.com",
            "https://www.facebook.com"]


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['first_round', 'second_round'],
    'q02': ['verify_child', 'missing_data_amounts'],
    'q03': ['cond_single_imputation'],
    'q04': ['quantitative_distribution', 'impute_height_quant'],
    'q05': ['answers']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """

    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" % (q, elt)
                raise Exception(stmt)

    return True
