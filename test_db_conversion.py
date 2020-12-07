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
from sqlalchemy import MetaData, Column, insert, Table
from nltk.tokenize import word_tokenize
import numpy as np

#%% Paths to the JSON files and import statements into dataframes

path_business = "data\yelp_archive\yelp_academic_dataset_business.json"
# path_user = "data\yelp_archive\yelp_academic_dataset_user.json"
# path_checkin = "data\yelp_archive\yelp_academic_dataset_checkin.json"

df_businesses = utils.read_json(path_business)
# df_users = utils.read_json(path_user)
# df_checkin = utils.read_json((path_checkin))
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
engine = create_engine('mysql+pymysql://%s:%s@localhost/yelp_challengedb' %(user_login, pword_login))

# This metadata object collects the schema from the existing db.
metadata = MetaData()
metadata.reflect(bind=engine)

# Confirmation of schema
tables_dict = metadata.tables


#%% Add data to the business table in the db and insert it with sqlalchemy.
# In practice, the df.to_sql() function seems more practical, and anecdotally seems faster.
connection = engine.connect()

# business = Table("business", metadata)
# ins = business.insert(values = )

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


# Add data to the business_attribute table in the db
df_business_attributes.to_sql('business_attributes', con=engine, if_exists='append', index=False)
#utils.insert_table(metadata, "business_attributes", engine, df_business_attributes)

#%% Tokenize the various categories to get all the business categories.

# Remove null entries
category_data = list(filter(None, list(df_businesses['categories'])))

# Split up each listed category and keep uniques.
category_token_list = [x.split(',') for x in category_data]
unique_tokens = set([x for sublist in category_token_list for x in sublist])

#%% Generate category ids, create df, and write to the db.
token_id = np.arange(len(unique_tokens), dtype = int)

df_category_ref = pd.DataFrame()
df_category_ref['category_id'] = token_id
df_category_ref['category_name'] = unique_tokens
df_category_ref.to_sql('category_ref', con=engine, if_exists='append', index=False)
