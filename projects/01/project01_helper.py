import os
import pandas as pd
import numpy as np

def get_category(grades, category, buffer):
    """
        get_category takes in a dataframe and the grade category, as well
        as the a string length buffer for the that specific category column.
        Returns a list of columns with all the assignment of the inputted
        category.

        :param grades: dataframe
        :param category: string name of the category we will look at
        :param buffer: int for the maximum length of the column name we want
        want to look at.
    """

    assignment = grades.filter(regex=category)
    mask = []
    for i in assignment.keys():
        if len(i) < buffer:
            mask.append(i)
    assignment = grades.filter(items=mask)
    return list(assignment.columns)

