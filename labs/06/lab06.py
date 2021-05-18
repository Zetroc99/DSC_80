import os
import pandas as pd
import numpy as np
import requests
import bs4
import json
from collections import deque


# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------


def question1():
    """
    NOTE: You do NOT need to do anything with this function.

    The function for this question makes sure you
    have a correctly named HTML file in the right
    place. Note: This does NOT check if the supplementary files
    needed for your page are there!

    >>> os.path.exists('lab06_1.html')
    True
    """

    # Don't change this function body!
    # No python required; create the HTML file.

    return


# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def extract_book_links(text):
    """
    :Example:
    >>> fp = os.path.join('data', 'products.html')
    >>> out = extract_book_links(open(fp, encoding='utf-8').read())
    >>> url = 'scarlet-the-lunar-chronicles-2_218/index.html'
    >>> out[1] == url
    True
    """
    def _rating_req(book):
        star_rec = ['four', 'five']
        if book.find('p').attrs['class'][1].lower() in star_rec:
            return True
        return False

    def _under_50(book):
        price_str = book.find('p', attrs={'class': 'price_color'}).text
        price_float = float("".join(filter(
            lambda x: x in '0123456789.', price_str)))
        if price_float < 50:
            return True
        return False

    def _find_url(book):
        return book.find('a').attrs['href']

    bs = bs4.BeautifulSoup(text, features='lxml')
    books = bs.find_all('article', attrs={'class': 'product_pod'})
    urls = []

    for book in books:
        if _rating_req(book) and _under_50(book):
            urls.append(_find_url(book))

    return urls


def get_product_info(text, categories):
    """
    :Example:
    >>> fp = os.path.join('data', 'Frankenstein.html')
    >>> out = get_product_info(open(fp, encoding='utf-8').read(), ['Default'])
    >>> isinstance(out, dict)
    True
    >>> 'Category' in out.keys()
    True
    >>> out['Rating']
    'Two'
    """
    bs = bs4.BeautifulSoup(text, features='lxml')
    category = bs.find('ul', attrs={'class': 'breadcrumb'}).find_all('a')[
        -1].text

    if category in categories:
        prod_info = bs.find('table',
                            attrs={'class': 'table table-striped'}).find_all(
            'td')
        row = {
            'Availability': prod_info[-2].text,
            'Category': category,
            'Description':
                bs.find('article', attrs={'class': 'product_page'}).find_all(
                    'p')[3].text,
            'Number of reviews': prod_info[-1].text,
            'Price (excl. tax)': prod_info[2].text,
            'Price (incl. tax)': prod_info[3].text,
            'Product Type': prod_info[1].text,
            'Rating':
                bs.find('p', attrs={'class': 'star-rating'}).attrs['class'][-1],
            'Tax': prod_info[4].text,
            'Title': bs.find('div', attrs={'class': 'product_main'}).find(
                'h1').text,
            'UPC': prod_info[0].text
        }
        return row
    return None


def scrape_books(k, categories):
    """
    :param k: number of book-listing pages to scrape.
    :returns: a dataframe of information on (certain) books
    on the k pages (as described in the question).

    :Example:
    >>> out = scrape_books(1, ['Mystery'])
    >>> out.shape
    (1, 11)
    >>> out['Rating'][0] == 'Four'
    True
    >>> out['Title'][0] == 'Sharp Objects'
    True
    """
    f = 'http://books.toscrape.com/'
    text = requests.get(f).text

    page_dicts = []

    for i in range(k):
        # search page w/ extract
        urls = extract_book_links(text)
        for url in urls:
            # get product info on page
            if i > 0:
                nextfp = os.path.join(f, 'catalogue/', url)
            else:
                nextfp = os.path.join(f, url)
            book = requests.get(nextfp).text
            info = get_product_info(book, categories)
            if info == None:
                continue
            page_dicts.append(info)

        bs = bs4.BeautifulSoup(text, 'lxml')
        next_page = bs.find('li', attrs={'class': 'next'}).find('a').attrs[
            'href']
        if i > 0:
            next_k = os.path.join(f, 'catalogue/', next_page)
        else:
            next_k = os.path.join(f, next_page)
        text = requests.get(next_k).text

    return pd.DataFrame(page_dicts)


# ---------------------------------------------------------------------
# Question 3
# ---------------------------------------------------------------------

def stock_history(ticker, year, month):
    """
    Given a stock code and month, return the stock price details for that month
    as a dataframe

    >>> history = stock_history('BYND', 2019, 6)
    >>> history.shape == (20, 13)
    True
    >>> history.label.iloc[-1]
    'June 03, 19'
    """
    end_date = pd.date_range(start=f'{year}/{month}/{1}', periods=1, freq='M')[
        0]
    stock_endpoint = f'https://financialmodelingprep.com/api/v3/historical' \
                     f'-price-full/{ticker}?from={year}-{month:02}-01&to=' \
                     f'{year}-{month:02}-{end_date.day:02}&apikey=fff40968' \
                     f'8b414e760842828d29517a91'
    response = requests.get(stock_endpoint).json()
    return pd.DataFrame(response['historical'])


def stock_stats(history):
    """
    Given a stock's trade history, return the percent change and transactions
    in billion dollars.

    >>> history = stock_history('BYND', 2019, 6)
    >>> stats = stock_stats(history)
    >>> len(stats[0]), len(stats[1])
    (7, 6)
    >>> float(stats[0][1:-1]) > 30
    True
    >>> float(stats[1][:-1]) > 1
    True
    """
    per_change = (
                (history.iloc[0]['close'] - history.iloc[-1][
                    'open']) / history.iloc[-1]['open'] * 100)
    if per_change > 0:
        per_change = str(per_change)
        per_change = '+' + per_change
    else:
        per_change = str(per_change)
        per_change = '-' + per_change
    p= per_change.split('.')
    percent = p[0] + '.' + p[1][:2] + '%'

    def transaction_vol(row):
        return ((row['low'] + row['high']) / 2) * row['volume']

    total = history.apply(transaction_vol, axis=1).sum() / 1_000_000_000
    total = str(total)
    t = total.split('.')
    ttv = t[0] + '.' + t[1][:2] + 'B'

    return percent, ttv


# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------

def get_comments(storyid):
    """
    Returns a dataframe of all the comments below a news story
    >>> out = get_comments(18344932)
    >>> out.shape
    (18, 5)
    >>> out.loc[5, 'by']
    'RobAtticus'
    >>> out.loc[5, 'time'].day
    31
    """
    news_endpoint = f"https://hacker-news.firebaseio.com/v0/item/{storyid}.json"
    response = requests.get(news_endpoint).json()
    comments = []

    def dfs(graph, data=None, q=None):
        if q is None:
            q = deque()
        if 'dead' in graph:
            q.append(graph)
        if 'kids' in graph:
            users = graph['kids']
            q.append(graph)
            for user_id in users:
                end_point = f"https://hacker-news.firebaseio.com/v0/item/{user_id}.json"
                new_graph = requests.get(end_point).json()
                dfs(new_graph, data, q)
                q.append(new_graph)
        if 'dead' not in graph:
            data.append(q.popleft())

    dfs(response, comments)
    df = pd.DataFrame(comments).iloc[1:].reset_index()\
        .filter(items=['id', 'by', 'parent', 'text', 'time'])
    df['time'] = df['time'].apply(lambda x: pd.to_datetime(x, unit='s'))

    return df


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['question1'],
    'q02': ['extract_book_links', 'get_product_info', 'scrape_books'],
    'q03': ['stock_history', 'stock_stats'],
    'q04': ['get_comments']
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
