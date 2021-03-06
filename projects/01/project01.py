import os
import pandas as pd
import numpy as np
from datetime import timedelta
import project01_helper as helper


# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    """
    get_assignment_names takes in a dataframe like grades and returns
    a dictionary with the following structure:

    The keys are the general areas of the syllabus: lab, project,
    midterm, final, disc, checkpoint

    The values are lists that contain the assignment names of that type.
    For example the lab assignments all have names of the form labXX where XX
    is a zero-padded two digit number. See the doctests for more details.

    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> names = get_assignment_names(grades)
    >>> set(names.keys()) == {'lab', 'project', 'midterm', 'final', 'disc', 'checkpoint'}
    True
    >>> names['final'] == ['Final']
    True
    >>> 'project02' in names['project']
    True
    """

    keys = ['lab', 'project', 'midterm', 'final', 'disc', 'checkpoint']
    vals = []

    vals.append(helper.get_category(grades, 'lab', 6))  # labs
    vals.append(helper.get_category(grades, 'project', 10))  # projects
    vals.append(helper.get_category(grades, 'Midterm', 10))  # midterms
    vals.append(helper.get_category(grades, 'Final', 6))  # finals
    vals.append(helper.get_category(grades, 'discussion', 13))  # discussions
    vals.append(helper.get_category(grades, 'checkpoint', 23))  # checkpoints

    return dict(zip(keys, vals))


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------


def projects_total(grades):
    """
    projects_total that takes in grades and computes the total project grade
    for the quarter according to the syllabus.
    The output Series should contain values between 0 and 1.

    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> out = projects_total(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """

    # get all project string names and begin total and point counts at 0
    projects = get_assignment_names(grades)['project']
    total = pd.Series(np.zeros(grades.shape[0]), index=grades.index)
    points = pd.Series(np.zeros(grades.shape[0]), index=grades.index)

    for project in projects:  # loop through each projectXX
        # get all columns associated with a projectXX
        proj_grades = grades.filter(regex=project).columns
        for string in proj_grades:  # loop through the col names
            if 'Max Points' in string and 'checkpoint' not in string:
                total += grades[string].fillna(0)
            elif string in projects:
                points += grades[string].fillna(0)
            elif 'free_response' in string and len(string) < 24:
                points += grades[string].fillna(0)

    return points / total


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------


def last_minute_submissions(grades):
    """
    last_minute_submissions takes in the dataframe 
    grades and a Series indexed by lab assignment that 
    contains the number of submissions that were turned 
    in on time by the student, yet marked 'late' by Gradescope.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = last_minute_submissions(grades)
    >>> isinstance(out, pd.Series)
    True
    >>> np.all(out.index == ['lab0%d' % d for d in range(1,10)])
    True
    >>> (out > 0).sum()
    8
    """

    labs = get_assignment_names(grades)['lab']
    threshold = timedelta(minutes=30, seconds=0, hours=2)
    due_time = timedelta(minutes=0, seconds=0, hours=0)
    errors = []

    for lab in labs:
        lab_grades = grades.filter(regex=lab).columns
        for string in lab_grades:
            if 'Lateness' in string:
                delta = pd.to_timedelta(grades[string])
                num_errors = \
                    grades.loc[(delta < threshold)].loc[
                        (delta > due_time)].shape[0]
                errors.append(num_errors)

    return pd.Series(errors, index=labs)


# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------

def lateness_penalty(col):
    """
    lateness_penalty takes in a 'lateness' column and returns 
    a column of penalties according to the syllabus.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> col = pd.read_csv(fp)['lab01 - Lateness (H:M:S)']
    >>> out = lateness_penalty(col)
    >>> isinstance(out, pd.Series)
    True
    >>> set(out.unique()) <= {1.0, 0.9, 0.7, 0.4}
    True
    """

    times = pd.to_timedelta(col)
    threshold = timedelta(minutes=30, seconds=0, hours=2)
    one_week_penalty = timedelta(weeks=1)
    two_week_penalty = timedelta(weeks=2)

    one_week_late = lambda \
            time: 0.9 if threshold < time < one_week_penalty else 1.0
    two_week_late = lambda \
            time: 0.7 if one_week_penalty < time < two_week_penalty else \
        one_week_late(time)
    very_late = lambda time: 0.4 if time > two_week_penalty else \
        two_week_late(time)

    return times.apply(very_late)


# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def process_labs(grades):
    """
    process_labs that takes in a dataframe like grades and returns
    a dataframe of processed lab scores. The output should:
      * share the same index as grades,
      * have columns given by the lab assignment names (e.g. lab01,...lab10)
      * have values representing the lab grades for each assignment, 
        adjusted for Lateness and scaled to a score between 0 and 1.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = process_labs(grades)
    >>> out.columns.tolist() == ['lab%02d' % x for x in range(1,10)]
    True
    >>> np.all((0.65 <= out.mean()) & (out.mean() <= 0.90))
    True
    """

    grades = grades.fillna(0)
    labs = get_assignment_names(grades)['lab']
    lab_scores = pd.DataFrame(columns=labs, index=grades.index)

    for lab in labs:
        lab_data = grades.filter(regex=lab)
        col = grades[lab].fillna(0)
        max_points = lab_data.filter(regex='Max').iloc[:, 0]
        scores = (lateness_penalty(col) * col) / max_points
        lab_scores[lab] = scores

    return lab_scores


# ---------------------------------------------------------------------
# Question #6
# ---------------------------------------------------------------------

def lab_total(processed):
    """
    lab_total takes in dataframe of processed assignments (like the output of 
    Question 5) and computes the total lab grade for each student according to
    the syllabus (returning a Series). 
    
    Your answers should be proportions between 0 and 1.

    :Example:
    >>> cols = 'lab01 lab02 lab03'.split()
    >>> processed = pd.DataFrame([[0.2, 0.90, 1.0]], index=[0], columns=cols)
    >>> np.isclose(lab_total(processed), 0.95).all()
    True
    """

    new_size = processed.shape[1] - 1
    total_scores = lambda student: (student.sum() - student.min()) / new_size

    return processed.apply(total_scores, axis=1)


# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

def total_points(grades):
    """
    total_points takes in grades and returns the final
    course grades according to the syllabus. Course grades
    should be proportions between zero and one.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """
    midterms = grades.filter(regex='Midterm')
    final = grades.filter(regex='Final')
    processed_labs = process_labs(grades)

    proj_total = projects_total(grades)
    labs_total = lab_total(processed_labs)
    midterm_total = midterms['Midterm'] / midterms['Midterm - Max Points']
    final_total = final['Final'] / final['Final - Max Points']
    disc_total = helper.assignment_total(grades, 'disc')
    cp_total = helper.assignment_total(grades, 'checkpoint')

    return pd.Series(labs_total * 0.2 + proj_total * 0.3 + cp_total * 0.025 +
                     disc_total * 0.025 + midterm_total * 0.15 +
                     final_total * 0.3, index=grades.index).fillna(0)


def final_grades(total):
    """
    final_grades takes in the final course grades
    as above and returns a Series of letter grades
    given by the standard cutoffs.

    :Example:
    >>> out = final_grades(pd.Series([0.92, 0.81, 0.41]))
    >>> np.all(out == ['A', 'B', 'F'])
    True
    """

    return total.apply(helper.grade_letter)


def letter_proportions(grades):
    """
    letter_proportions takes in the dataframe grades 
    and outputs a Series that contains the proportion
    of the class that received each grade.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = letter_proportions(grades)
    >>> np.all(out.index == ['B', 'C', 'A', 'D', 'F'])
    True
    >>> out.sum() == 1.0
    True
    """
    letter_grades = final_grades(total_points(grades))

    return letter_grades.value_counts() / grades.shape[0]


# ---------------------------------------------------------------------
# Question # 8
# ---------------------------------------------------------------------

def simulate_pval(grades, N):
    """
    simulate_pval takes in the number of
    simulations N and grades and returns
    the likelihood that the grade of seniors
    was worse than the class under null hypothesis conditions
    (i.e. calculate the p-value).

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = simulate_pval(grades, 100)
    >>> 0 <= out <= 0.1
    True
    """
    SR_grades = grades[grades['Level'] == 'SR']
    stats = np.array([])
    obs_stat = np.round(total_points(SR_grades).mean(), 2)
    total = total_points(grades)

    for _ in range(N):
        test_stat = np.round(total.sample(grades.shape[0]).mean(), 2)
        stats = np.append(stats, test_stat)
    p_val = np.count_nonzero(stats < obs_stat) / N

    return p_val


# ---------------------------------------------------------------------
# Question # 9
# ---------------------------------------------------------------------


def total_points_with_noise(grades):
    """
    total_points_with_noise takes in a dataframe like grades, 
    adds noise to the assignments as described in notebook, and returns
    the total scores of each student calculated with noisy grades.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points_with_noise(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """
    midterms = grades.filter(regex='Midterm')
    final = grades.filter(regex='Final')
    processed_labs = process_labs(grades)
    noise = np.random.normal(0, 0.02, size=grades.shape[0])

    proj_total = projects_total(grades) + noise
    labs_total = lab_total(processed_labs) + noise
    midterm_total = (midterms['Midterm'] / midterms[
        'Midterm - Max Points']) + noise
    final_total = (final['Final'] / final['Final - Max Points']) + noise
    disc_total = helper.assignment_total(grades, 'disc') + noise
    cp_total = helper.assignment_total(grades, 'checkpoint') + noise

    return np.clip(pd.Series(labs_total * 0.2 + proj_total * 0.3 + cp_total *
                             0.025 + disc_total * 0.025 + midterm_total * 0.15 +
                             final_total * 0.3, index=grades.index).fillna(0),
                   0, 1)


# ---------------------------------------------------------------------
# Question #10
# ---------------------------------------------------------------------

def short_answer():
    """
    short_answer returns (hard-coded) answers to the 
    questions listed in the notebook. The answers should be
    given in a list with the same order as questions.

    :Example:
    >>> out = short_answer()
    >>> len(out) == 5
    True
    >>> len(out[2]) == 2
    True
    >>> 50 < out[2][0] < 100
    True
    >>> 0 < out[3] < 1
    True
    >>> isinstance(out[4][0], bool)
    True
    >>> isinstance(out[4][1], bool)
    True
    """

    return [0.015, 59.81, [57.38, 65.23], 0.166, [False, False]]


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_assignment_names'],
    'q02': ['projects_total'],
    'q03': ['last_minute_submissions'],
    'q04': ['lateness_penalty'],
    'q05': ['process_labs'],
    'q06': ['lab_total'],
    'q07': ['total_points', 'final_grades', 'letter_proportions'],
    'q08': ['simulate_pval'],
    'q09': ['total_points_with_noise'],
    'q10': ['short_answer']
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
