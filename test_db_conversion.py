import pandas as pd
import json
import mysql.connector
from dotenv import load_dotenv, find_dotenv
import os
import utils
load_dotenv()
from importlib import reload
from mysql.connector import errorcode


#%% This simple setup code was taken from https://www.kaggle.com/vksbhandary/exploring-yelp-reviews-dataset`
def init_ds(json):
    ds = {}
    keys = json.keys()
    for k in keys:
        ds[k] = []
    return ds, keys


def read_json(file):
    dataset = {}
    keys = []
    with open(file, encoding='utf8') as file_lines:
        for count, line in enumerate(file_lines):
            data = json.loads(line.strip())
            if count == 0:
                dataset, keys = init_ds(data)
            for k in keys:
                dataset[k].append(data[k])

        return pd.DataFrame(dataset)
#%%
path_business = "data\yelp_archive\yelp_academic_dataset_business.json"

yelp_business = read_json(path_business)

#%% Test opening db

user_login = os.environ['db_login']
pword_login = os.environ['db_pword']

cnx = mysql.connector.connect(user = user_login , password = pword_login,
                              host = '127.0.0.1' , database = 'yelp_challengedb')
cnx.close()

#%% Table creation from within python

DB_NAME = 'test_db'

TABLES = {}

TABLES['business'] = (
	" CREATE TABLE `business` ("
	" `business_id` char(22) NOT NULL,"
	" `name` varchar(50) NOT NULL,"
	" `address` varchar(100) NOT NULL,"
	" `state` varchar(20),"
	" `city` varchar(20),"
	" `is_open` int,"
	" `latitude` int,"
	" `longitude` int,"
	" `postal_code` int,"
	" `type_id` int,"
	" PRIMARY KEY (`business_id`)" 
	" ) ENGINE = InnoDB"
	)
#%%
# Tries to create a db if it does not exist
cnx = mysql.connector.connect(user=user_login, password = pword_login)
cursor = cnx.cursor()
#%%
try:
    cursor.execute("USE {}".format(DB_NAME))
except mysql.connector.Error as err:
    print("Database {} does not exists.".format(DB_NAME))
    if err.errno == errorcode.ER_BAD_DB_ERROR:
        utils.create_database(cursor, DB_NAME)
        print("Database {} created successfully.".format(DB_NAME))
        cnx.database = DB_NAME
    else:
        print(err)
        exit(1)

#%% Add a table

cursor.execute("USE {}".format(DB_NAME))
for table_name in TABLES:
    table_description = TABLES[table_name]
    try:
        print("Creating table {}: ".format(table_name), end='')
        cursor.execute(table_description)
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
            print("already exists.")
        else:
            print(err.msg)
    else:
        print("OK")

cursor.close()
cnx.close()