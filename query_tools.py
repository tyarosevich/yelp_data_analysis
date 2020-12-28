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
#%%
# Login info for my local MySQL db, stored in a .env file.
user_login = os.environ['db_login']
pword_login = os.environ['db_pword']
# Creates a sqlalchmy engine for use throughout the project.
engine = create_engine('mysql+pymysql://%s:%s@localhost/yelp_challengedb' %(user_login, pword_login), pool_recycle=3600, pool_size=5)

#%% Collect specific id/city info from 'business' and business_id with stars from 'review' in order
# to query the star breakdown of a given city.
city = 'Scottsdale'
query_str1 = 'SELECT business_id, city FROM business WHERE city = "{}"'.format(city)
query_str2 = 'SELECT business_id, stars FROM review'

df_business_city = pd.read_sql(query_str1, engine)

df_review_stars = pd.read_sql(query_str2, engine)

#%%
city_id_set = set(df_business_city['business_id'])

# TRY THIS WITH ITERTUPLES
# city_business = [(row[0], row[1]) for i, row in df_review_stars.iterrows() if row[0] in city_id_set]

# Hash error cause series.
df_city_business = df_review_stars[df_review_stars['business_id'] in city_id_set]

#%% Create dataframe comprised of reviews from the given city.

df_city_reviews = pd.DataFrame(city_business, columns = ['business_id', 'stars'])
