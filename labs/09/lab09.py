import pandas as pd
import numpy as np
import seaborn as sns
import os

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Binarizer

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------


def simple_pipeline(data):
    """
    simple_pipeline takes in a dataframe like data and returns a tuple
    consisting of the pipeline and the predictions your model makes
    on data (as trained on data).

    :Example:
    >>> fp = os.path.join('data', 'toy.csv')
    >>> data = pd.read_csv(fp)
    >>> pl, preds = simple_pipeline(data)
    >>> isinstance(pl, Pipeline)
    True
    >>> isinstance(pl.steps[-1][1], LinearRegression)
    True
    >>> isinstance(pl.steps[0][1], FunctionTransformer)
    True
    >>> preds.shape[0] == data.shape[0]
    True
    """
    pl = Pipeline(
        [('logScale', FunctionTransformer(np.log)), ('lr', LinearRegression())])

    return pl.fit(data[['c2']], data['y']), pl.predict(data[['c2']])


# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def multi_type_pipeline(data):
    """
    multi_type_pipeline that takes in a dataframe like data and
    returns a tuple consisting of the pipeline and the predictions
    your model makes on data (as trained on data).

    :Example:
    >>> fp = os.path.join('data', 'toy.csv')
    >>> data = pd.read_csv(fp)
    >>> pl, preds = multi_type_pipeline(data)
    >>> isinstance(pl, Pipeline)
    True
    >>> isinstance(pl.steps[-1][1], LinearRegression)
    True
    >>> isinstance(pl.steps[0][1], ColumnTransformer)
    True
    >>> data.shape[0] == preds.shape[0]
    True
    """
    num_feat = ['c2']
    cat_feat = ['group']

    pp = ColumnTransformer(
        transformers=[
            ('num', FunctionTransformer(np.log), num_feat),
            ('cat', OneHotEncoder(), cat_feat)
        ], remainder='passthrough')

    pl = Pipeline(
        steps=[('preprocess', pp), ('regression', LinearRegression())])
    pl.fit(data.drop('y', axis=1), data.y)

    return pl.fit(data.drop('y', axis=1), data.y), pl.predict(data.drop(
        'y', axis=1))


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

from sklearn.base import BaseEstimator, TransformerMixin


class StdScalerByGroup(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        :Example:
        >>> cols = {'g': ['A', 'A', 'B', 'B'], 'c1': [1, 2, 2, 2], 'c2': [3, 1, 2, 0]}
        >>> X = pd.DataFrame(cols)
        >>> std = StdScalerByGroup().fit(X)
        >>> std.grps_ is not None
        True
        """
        # X may not be a pandas dataframe (e.g. a np.array)
        df = pd.DataFrame(X)
        df_groups = df[df.columns[0]].sort_values().unique()

        df_group_mean = df.groupby(df.columns[0]).mean()
        df_group_std = df.groupby(df.columns[0]).std()

        # A dictionary of means/standard-deviations for each column,
        # for each group.
        self.grps_ = df.groupby(df.columns[0]).agg(['mean', 'std']).to_dict()

        return self

    def transform(self, X, y=None):
        """
        :Example:
        >>> cols = {'g': ['A', 'A', 'B', 'B'], 'c1': [1, 2, 3, 4], 'c2': [1, 2, 3, 4]}
        >>> X = pd.DataFrame(cols)
        >>> std = StdScalerByGroup().fit(X)
        >>> out = std.transform(X)
        >>> out.shape == (4, 2)
        True
        >>> np.isclose(out.abs(), 0.707107, atol=0.001).all().all()
        True
        """

        try:
            getattr(self, "grps_")
        except AttributeError:
            raise RuntimeError(
                "You must fit the transformer before transforming the data!")

        # Define a helper function here?
        def transform_helper(x):
            return (x-x.mean()) / x.std()
        # X may not be a dataframe (e.g. np.array)
        df = pd.DataFrame(X)

        return df.groupby(df.columns[0]).transform(transform_helper)

# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------

def eval_toy_model():
    """
    hardcoded answers to question 4

    :Example:
    >>> out = eval_toy_model()
    >>> len(out) == 3
    True
    """

    return [0.3956, 0.5733, 0.5730]


# ---------------------------------------------------------------------
# Question # 5
# ---------------------------------------------------------------------

def tree_reg_perf(galton):
    """

    :Example:
    >>> galton_fp = os.path.join('data', 'galton.csv')
    >>> galton = pd.read_csv(galton_fp)
    >>> out = tree_reg_perf(galton)
    >>> out.columns.tolist() == ['train_err', 'test_err']
    True
    >>> out['train_err'].iloc[-1] < out['test_err'].iloc[-1]
    True
    """
    X = galton.drop('childHeight', axis=1)
    y = galton.childHeight

    rmse_train = []
    rmse_test = []
    for depth in range(1, 21):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.30)
        treeReg = DecisionTreeRegressor(max_depth=depth).fit(X_train, y_train)
        pred_train = treeReg.predict(X_train)
        pred_test = treeReg.predict(X_test)

        err_train = (np.sum((y_train - pred_train) ** 2) / len(y_train)) ** 0.5
        rmse_train.append(err_train)

        err_test = (np.sum((y_test - pred_test) ** 2) / len(y_test)) ** 0.5
        rmse_test.append(err_test)

    return pd.DataFrame(index=np.arange(1, 21), data={'train_err': rmse_train,
                                                      'test_err': rmse_test})


def knn_reg_perf(galton):
    """
    :Example:
    >>> galton_fp = os.path.join('data', 'galton.csv')
    >>> galton = pd.read_csv(galton_fp)
    >>> out = knn_reg_perf(galton)
    >>> out.columns.tolist() == ['train_err', 'test_err']
    True
    """
    X = galton.drop('childHeight', axis=1)
    y = galton.childHeight

    rmse_train = []
    rmse_test = []
    for n in range(1, 21):
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y, test_size=0.30)
        knnReg = KNeighborsRegressor(n_neighbors=n).fit(X_train, y_train)
        pred_train = knnReg.predict(X_train)
        pred_test = knnReg.predict(X_test)

        err_train = (np.sum((y_train - pred_train) ** 2) / len(y_train)) ** 0.5
        rmse_train.append(err_train)

        err_test = (np.sum((y_test - pred_test) ** 2) / len(y_test)) ** 0.5
        rmse_test.append(err_test)

    return pd.DataFrame(index=np.arange(1, 21), data={'train_err': rmse_train,
                                                      'test_err': rmse_test})


# ---------------------------------------------------------------------
# Question # 6
# ---------------------------------------------------------------------

def titanic_model(titanic):
    """
    :Example:
    >>> fp = os.path.join('data', 'titanic.csv')
    >>> data = pd.read_csv(fp)
    >>> pl = titanic_model(data)
    >>> isinstance(pl, Pipeline)
    True
    >>> from sklearn.base import BaseEstimator
    >>> isinstance(pl.steps[-1][-1], BaseEstimator)
    True
    >>> preds = pl.predict(data.drop('Survived', axis=1))
    >>> ((preds == 0)|(preds == 1)).all()
    True
    """
    titanic = titanic.copy()
    titanic['Name'] = titanic.Name.map(lambda x: x.split('.')[0].strip())
    titanic['Name'] = titanic['Name'].apply(
        lambda x: 1 if 'Mrs' in x or 'Miss' in x else 0)

    X = titanic.drop('Survived', axis=1)
    y = titanic.Survived

    class_group = ['Pclass', 'Age', 'Fare']
    cat_feat = list(titanic.select_dtypes(include='object').columns)

    ct = ColumnTransformer([
        ('class', StdScalerByGroup(), class_group),
        ('categories', OneHotEncoder(), cat_feat)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    params = {'KNN__n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
    clf = Pipeline(
        steps=[('colTrans', ct), ('grid', KNeighborsClassifier(n_neighbors=9))])

    return clf.fit(X_train, y_train)


# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


def json_reader(file, iterations):
    """
    :Example
    >>> fp = os.path.join('data', 'reviews.json')
    >>> reviews, labels = json_reader(fp, 5000)
    >>> isinstance(reviews, list)
    True
    >>> isinstance(labels, list)
    True
    >>> len(labels) == len(reviews)
    True
    """
    reviews = pd.read_json(file, lines=True).head(iterations)
    reviewsText = reviews.reviewText.tolist()
    rating = reviews.overall.tolist()

    return reviewsText, rating


def create_classifier_multi(X, y):
    """
    :Example
    >>> fp = os.path.join('data', 'reviews.json')
    >>> reviews, labels = json_reader(fp, 5000)
    >>> trial = create_classifier_multi(reviews, labels)
    >>> isinstance(trial, Pipeline)
    True
    """

    return ...


def to_binary(labels):
    """
    :Example
    >>> lst = [1, 2, 3, 4, 5]
    >>> to_binary(lst)
    >>> print(lst)
    [0, 0, 0, 1, 1]
    """

    return ...


def create_classifier_binary(X, y):
    """
    :Example
    >>> fp = os.path.join('data', 'reviews.json')
    >>> reviews, labels = json_reader(fp, 5000)
    >>> to_binary(labels)
    >>> trial = create_classifier_multi(reviews, labels)
    >>> isinstance(trial, Pipeline)
    True
    """

    return ...


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['simple_pipeline'],
    'q02': ['multi_type_pipeline'],
    'q03': ['StdScalerByGroup'],
    'q04': ['eval_toy_model'],
    'q05': ['tree_reg_perf', 'knn_reg_perf'],
    'q06': ['titanic_model']
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
