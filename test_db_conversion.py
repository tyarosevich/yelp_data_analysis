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


#%%
path_business = "data\yelp_archive\yelp_academic_dataset_business.json"
# path_user = "data\yelp_archive\yelp_academic_dataset_user.json"
# path_checkin = "data\yelp_archive\yelp_academic_dataset_checkin.json"

df_businesses = utils.read_json(path_business)
# df_users = utils.read_json(path_user)
# df_checkin = utils.read_json((path_checkin))
#%% Test opening db

user_login = os.environ['db_login']
pword_login = os.environ['db_pword']

cnx = mysql.connector.connect(user = user_login , password = pword_login,
                              host = '127.0.0.1' , database = 'yelp_challengedb')
cnx.close()



#%% Create metadata object with schema from existing db

engine = create_engine('mysql+pymysql://%s:%s@localhost/yelp_challengedb' %(user_login, pword_login))
metadata = MetaData()
metadata.reflect(bind=engine)

# Confirmation of schema
tables_dict = metadata.tables


#%% Add data to the business table in the db
business = Table("business", metadata)
connection = engine.connect()

ins = business.insert()



#%% Making sub dataframe
sub_bus_frame = df_businesses.filter(['business_id', 'name', 'address', 'city', 'stated', 'postal_code', 'latitude', 'longitude', 'is_open']).copy()

#%% Actually a list of dicts, each dict is a row of the dataframe where the keys are the column names, values are row values.
test_dict = sub_bus_frame.to_dict('records')

#%% Worked. The default columns are written to the db (but not the columns that need to be distributed).
connection.execute(ins, test_dict)

#%% Get unique business attributes
attribute_list = list(df_businesses['attributes'])

#%% Get unique keys
# filt_list = filter(None, attribute_list)
# unique_attr = list(set(key for dict in filt_list for key in dict.keys()))

key_list = ['BusinessAcceptsBitcoin', 'BusinessAcceptsCreditCards', 'DogsAllowed', 'WheelchairAccessible', 'AcceptsInsurance',
            'Open24Hours', 'Corkage', 'BikeParking', 'HasTV', 'RestaurantsDelivery', 'Music',
            'Smoking', 'HappyHour', 'GoodForKids', 'RestaurantsTakeOut', 'OutdoorSeating', 'DriveThru', 'CoatCheck']
#%% Test list comprehensions
attribute = 'BusinessAcceptsCreditCards'
attribute_list_noempty = ['' if v is None else v for v in attribute_list]
test_output = [x[attribute] if attribute in x else False for x in attribute_list_noempty]

#%% Build the business_attribute dataframe

df_business_attributes = pd.DataFrame()
for attribute in key_list:
    df_business_attributes[attribute] = [x[attribute] if attribute in x else False for x in attribute_list_noempty]

# #%% Clean any non-boolean values
# shape = df_business_attributes.shape
#
# for i in range(shape[0]):
#     for j in range(shape[1]):
#         if isinstance(df_business_attributes.iat[i,j], bool):
#             pass
#         else:
#             df_business_attributes.iat[i,j] = False
#%% Convert string booleans to booleans
for c in df_business_attributes.columns:
    df_business_attributes[c] = [x=="True" if type(x) == str else x for x in df_business_attributes[c]]

#%% Add primary keys
df_business_attributes['business_id'] = df_businesses['business_id']
#%% Adjust headers to match db
df_business_attributes.rename(columns = {'BusinessAcceptsBitcoin': 'businessAcceptsBitcoin','BusinessAcceptsCreditCards': 'businessAcceptsCreditCards', 'DogsAllowed':'dogsAllowed',
                               'WheelchairAccessible':'wheelchairAccessible'}, inplace = True)


#%% Add data to the business_attribute table in the db

utils.insert_table(metadata, "business_attributes", engine, df_business_attributes)

