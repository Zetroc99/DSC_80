import os
import pandas as pd
import numpy as np
import project01 as proj


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


def assignment_total(grades, assignment):
    """
    assignment_total that takes in grades and computes the total assignment
    grade for the quarter according to the syllabus.
    The output Series should contain values between 0 and 1.

    :Example:
    """

    # get all assignment string names and begin total and point counts at 0
    assignments = proj.get_assignment_names(grades)[assignment]
    total = pd.Series(np.zeros(grades.shape[0]), index=grades.index)
    points = pd.Series(np.zeros(grades.shape[0]), index=grades.index)

    for assignment in assignments:  # loop through each assignment
        # get all columns associated with a assignment
        assignments_grades = grades.filter(regex=assignment).columns
        for string in assignments_grades:  # loop through the col names
            if 'Max Points' in string:
                total += grades[string].fillna(0)
            elif string in assignments:
                points += grades[string].fillna(0)

    return points / total


def grade_letter(grade):
    if grade >= 0.9:
        return 'A'
    elif 0.9 > grade >= 0.8:
        return 'B'
    elif 0.8 > grade >= 0.7:
        return 'C'
    elif 0.7 > grade >= 0.6:
        return 'D'
    else:
        return 'F'
