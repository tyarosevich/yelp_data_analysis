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


#%%
business = Table("business", metadata)
connection = engine.connect()

ins = business.insert()

connection.execute(ins, business_id = '23kj2l3k4jlk23', name = 'farts r us', address = '31 farts lane')

#%% Making sub dataframe
sub_bus_frame = df_businesses.filter(['business_id', 'name', 'address', 'city', 'stated', 'postal_code', 'latitude', 'longitude', 'is_open']).copy()
#%% Actually a list of dicts, each dict is a row of the dataframe where the keys are the column names, values are row values.
test_dict = sub_bus_frame.to_dict('records')
#%% Worked. The default columns are written to the db (but not the columns that need to be distributed).
connection.execute(ins, test_dict)


