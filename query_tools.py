import utils
import pandas as pd
import numpy as np
from sqlalchemy import MetaData, Column, insert, Table
import pymysql
from mysql.connector import errorcode
from sqlalchemy import create_engine
import mysql.connector
from dotenv import load_dotenv, find_dotenv
load_dotenv()
import os
import timeit
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from importlib import reload

#%%
# Login info for my local MySQL db, stored in a .env file.
user_login = os.environ['db_login']
pword_login = os.environ['db_pword']
# Creates a sqlalchmy engine for use throughout the project.
engine = create_engine('mysql+pymysql://%s:%s@localhost/yelp_challengedb' %(user_login, pword_login), pool_recycle=3600, pool_size=5)

#%%
def city_review_totals(engine, city):
    '''
    Queries the local database to get review totals by city.
    :param engine: Engine
        A sqlalchemy engine to connect to the DB.
    :param city: str
        Name of the city to be queried.
    :return: DataFrame
        A dataframe containing the totals.
    '''
    # Queries to pull city-specific info from the db.
    query_str1 = 'SELECT business_id, city FROM business WHERE city = "{}"'.format(city)
    query_str2 = 'SELECT business_id, stars FROM review'

    # Reads in to dataframes from the db.
    df_business_city = pd.read_sql(query_str1, engine)
    df_review_stars = pd.read_sql(query_str2, engine)

    # Hashable to check whether a review is from the city.
    city_id_set = set(df_business_city['business_id'])

    # List comprehension to keep only reviews from the city. Note, vectorization is not possible because
    # Series are not hashable (Not sure if there is a workaround).
    city_business = [(row[0], row[1]) for row in df_review_stars.itertuples(index=False) if row[0] in city_id_set]

    # Form into dataframe and count value totals
    df_city_reviews = pd.DataFrame(city_business, columns=['business_id', 'stars'])
    df_review_totals = df_city_reviews['stars'].value_counts().to_frame().reset_index().rename(
    columns={'index': 'stars', 'stars': 'total_count'}).sort_values('stars', axis=0)

    # Plot the result
    %matplotlib
    sns.set_style("dark")
    sns.barplot(data=df_review_totals, x='stars', y='total_count')
    plt.title("Review Totals for City of {}".format(city))
    plt.show()
    return df_review_totals

#%% Query review totals for a city.
city = 'Toronto'
df_test = city_review_totals(engine, city)

#%% Top 10 of given label.

tag = 'restaurants'

def top_ten_tag(tag, engine):
    '''
    Retrieves the total counts for a given category tag in all cities and plots the top 10.
    :param tag: str
        The tag to search for.
    :param engine: Engine
        sqlalchemy engine to query DB
    :return: DataFrame
        Dataframe containing the full list of cities with tag totals.
    '''
    query1 = 'SELECT * FROM category_ref WHERE LOWER(category_name) LIKE "{}"'.format(tag)
    df_tag = pd.read_sql(query1, engine)

    if len(df_tag['category_id']) == 0:
        raise ValueError('There are no matching tags')
        return

    tag_id = df_tag['category_id'][0]

    # Query to retrieve totals for given tag from all cities.
    query2 = (  'SELECT b.city, '
                'COUNT(city) as cnt '
                'FROM business b '
                'INNER JOIN business_category bc '
                'ON bc.business_id=b.business_id WHERE category_id={} '
                'GROUP BY b.city '
                'ORDER BY cnt DESC; '.format(tag_id)
    )

    # Retrieve the data
    df_totals = pd.read_sql(query2, engine)

    # Plot the top 10
    %matplotlib
    sns.set_style("dark")
    sns.barplot(data=df_totals[0:10], y='city', x='cnt', palette='Blues_d')
    plt.title('Total number of businesses in {} with of the category "{}"'.format(city, tag))
    plt.show()
    return df_totals

#%% Testing

df_top_seafood = top_ten_tag('seafood', engine)

#%% Geospatial PCA

# Get a large sample of geospatial data and stars

query = 'SELECT latitude, longitude, stars FROM business LIMIT 10000'
df_geo_dat = pd.read_sql(query, engine)

#%%
mat_geo = normalize(df_geo_dat.to_numpy().T, axis=0, norm='l1')
#%%
U, S, VH = np.linalg.svd(mat_geo)
#%%
# Singular value proportions
sum = np.sum(S)
sv_ratios=S/sum
ax = sns.barplot(x=['$\sigma_1$', '$\sigma_2$', '$\sigma_3$'], y = sv_ratios, palette='Blues_d')
plt.title('Proportion of variance captured by singular value')
utils.change_width(ax, .35)
plt.show()