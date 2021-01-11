import pandas as pd
import json
import mysql.connector
from dotenv import load_dotenv, find_dotenv
import os
import utils
load_dotenv()
from importlib import reload
from mysql.connector import errorcode
from sqlalchemy import create_engine
import pymysql
from sqlalchemy import MetaData, Column, insert, Table, update
from nltk.tokenize import word_tokenize
import numpy as np
import gc
from sys import getsizeof

#%% Paths to the JSON files and import statements into dataframes

path_business = "data\yelp_archive\yelp_academic_dataset_business.json"
path_user = "data\yelp_archive\yelp_academic_dataset_user.json"
path_checkin = "data\yelp_archive\yelp_academic_dataset_checkin.json"
path_review = "data\yelp_archive\yelp_academic_dataset_review.json"

df_businesses = utils.read_json(path_business)

#%% Test opening db

# Login info for my local MySQL db, stored in a .env file.
user_login = os.environ['db_login']
pword_login = os.environ['db_pword']

# Check the connection.
cnx = mysql.connector.connect(user = user_login , password = pword_login,
                              host = '127.0.0.1' , database = 'yelp_challengedb')
cnx.close()


#%% Create metadata object with schema from existing db

# Creates a sqlalchmy engine for use throughout the project.
engine = create_engine('mysql+pymysql://%s:%s@localhost/yelp_challengedb' %(user_login, pword_login), pool_recycle=3600, pool_size=5)

#%%
df_businesses.to_csv(path_or_buf="temp_table.csv", columns=['business_id', 'latitude', 'longitude'], float_format='%.8f', index=False)

#%% Create a sub-dataframe to migrate the JSON info that already fits the schema.
# Essentially, splits the data off that we will need to reformat for the schema,
# and commits the data that's already formatted correctly.
sub_bus_frame = df_businesses.filter(['business_id', 'name', 'address', 'city', 'state', 'postal_code', 'latitude', 'longitude', 'stars', 'review_count', 'is_open']).copy()

# Writes to the db using the sqlalchemy engine.
sub_bus_frame.to_sql('business', con=engine, if_exists='append', index=False)

#%% Some pre-processing of the attributes to fit the db schema.
# Collect the column of business attributes.
attribute_list = list(df_businesses['attributes'])

#%% Get unique keys
# filt_list = filter(None, attribute_list)
# unique_attr = list(set(key for dict in filt_list for key in dict.keys()))

#%% Manually adjusted attribute list. Many were not in a boolean format, and for simplicity they
# have been discarded. In practice, the non-boolean attributes could have been parsed out and simply added
# to the attributes table if they were desired for analysis.
attribute_list_noempty = ['' if v is None else v for v in attribute_list]
key_list = ['BusinessAcceptsBitcoin', 'BusinessAcceptsCreditCards', 'DogsAllowed', 'WheelchairAccessible', 'AcceptsInsurance',
            'Open24Hours', 'Corkage', 'BikeParking', 'HasTV', 'RestaurantsDelivery', 'Music',
            'Smoking', 'HappyHour', 'GoodForKids', 'RestaurantsTakeOut', 'OutdoorSeating', 'DriveThru', 'CoatCheck']

#%% Build the business_attribute dataframe

df_business_attributes = pd.DataFrame()
for attribute in key_list:
    df_business_attributes[attribute] = [x[attribute] if attribute in x else False for x in attribute_list_noempty]

# Convert string booleans to boolean type.
for c in df_business_attributes.columns:
    df_business_attributes[c] = [x=="True" if type(x) == str else x for x in df_business_attributes[c]]

#%% Add primary keys
df_business_attributes['business_id'] = df_businesses['business_id']

# Adjust headers to match db
df_business_attributes.rename(columns = {'BusinessAcceptsBitcoin': 'businessAcceptsBitcoin','BusinessAcceptsCreditCards': 'businessAcceptsCreditCards', 'DogsAllowed':'dogsAllowed',
                               'WheelchairAccessible':'wheelchairAccessible'}, inplace = True)

#%%
# Add data to the business_attribute table in the db
df_business_attributes.to_sql('business_attributes', con=engine, if_exists='append', index=False)
#utils.insert_table(metadata, "business_attributes", engine, df_business_attributes)

#%% Tokenize the various categories to get all the business categories.

# Remove null entries
category_data = list(filter(None, list(df_businesses['categories'])))

# Split up each listed category and keep uniques.
category_token_list = [x.split(',') for x in category_data]
unique_tokens = set([x.strip() for sublist in category_token_list for x in sublist])

#%% Generate category ids, create df, and write to csv for DB updating.
token_id = np.arange(len(unique_tokens), dtype = int)

df_category_ref = pd.DataFrame()
df_category_ref['category_id'] = token_id
df_category_ref['category_name'] = unique_tokens

#%%
df_category_ref.to_csv(path_or_buf='temp_cat_ref.csv', columns=['category_id', 'category_name'], index=False)

#%% Make a key value dict by name:id for the categories.
cat_names = list(unique_tokens)
cat_id = list(token_id)
category_keys = {cat_names[i]:cat_id[i] for i in range(len(cat_id))}
#%% Create the business_category data frame


cat_business_setup_list = []

# Iterates through the category strings, checks for membership
# and creates a one-to-one entry for that business and category.
# Possible faster solutions with complex nested list comprehension
# but this one was sub one minute for 200k rows.
for index, row in df_businesses.iterrows():
    if row['categories'] == None:
        continue
    for cat in row['categories'].split(','):
        cat = cat.strip()
        if cat in unique_tokens:
            cat_business_setup_list.append( [row['business_id'], category_keys[cat]])

# Write the data frame from list of lists.
headers = ['business_id', 'category_id']
df_business_category = pd.DataFrame(cat_business_setup_list, columns=headers)

# Write to csv to update the db. Way too big to write in with sqlalchemy (takes a long time).
df_business_category.to_csv(path_or_buf='temp_bus_cat.csv', columns=['business_id', 'category_id'], index=False)

#%% Separate friend column to different table.

df_friend_relations = df_users.filter(['user_id', 'friends']).copy()
df_users.drop('friends', axis=1, inplace=True)

#%% Write to the db
pd.to_datetime(df_users.yelping_since)

# Drop elite, whatever that is.
df_users.drop('elite', axis=1, inplace=True)

df_users.to_sql('users', con=engine, if_exists='append', index=False)

#%% Creates the relationships dataframe by creating a list of nested lists.
# Each nested list is a row of the dataframe. Note this table is quite large (204 million elements).

df_relationships = pd.DataFrame(
    [ [user1, user2] for user1, fr_list in df_friend_relations.itertuples(index=False, name=None) for user2 in fr_list.split(',')]
)

df_relationships.columns = ['user1_id', 'user2_id']

#%% Release stuff from memory
del [[df_users, df_friend_relations]]
gc.collect()
Out[10]: 15
df_users=pd.DataFrame()
df_friend_relations=pd.DataFrame()

#%% Write relations to db
df_relationships.to_sql('relationships', con=engine, if_exists='append', index=False, chunksize=10000)


#%% Localize datetime values
df_review['date'] = pd.to_datetime(df_review['date']).dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
#%% Write reviews to db
df_review.to_sql('review', con=engine, if_exists='append', index=False, chunksize=2000, method='multi')



