import mysql.connector
from mysql.connector import errorcode
import json
import pandas as pd
from sqlalchemy import create_engine
import pymysql
from sqlalchemy import MetaData, Column, insert, Table


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

# Tries to create a database if it doesn't exist
def create_database(cursor, DB_NAME):
    try:
        cursor.execute(
            "CREATE DATABASE {} DEFAULT CHARACTER SET 'utf8'".format(DB_NAME))
    except mysql.connector.Error as err:
        print("Failed creating database: {}".format(err))
        exit(1)

def table_string_constructor(table_name, columns_dict, prim_key, constraint_dict):
    '''
    A function to create SQL code to be passed via mysqlconnector to create tables.
    :param table_name: str
        Name of the table to be created.
    :param columns_dict: dict
        Format is str:tuple. Str is name of the column, tuple is (type, Boolean).
        The boolean determines whether it is NOT NULL. True = NOT NULL.
    :param prim_key: str
        String of the primary key.
    :param constraint_dict: dict
        Format is constraint name : tuple. The tuple is (key name, target table, target key, Boolean).
        The Boolean determines whether or not there is DELETE CASCADE. True = DELETE CASCADE.
    :return:
    '''

    # Initialize string with table name.
    string = 'CREATE TABLE `%s` (' %(table_name)

    # Loop through each intended column name and type with boolean
    # for the presence of NOT NULL.
    for name, tup in columns_dict.items():
        if tup[1]:
            string += ' `%s` %s NOT NULL,' %(name, tup[0])
        else:
            string += ' `%s` %s,' %(name, tup[0])

    # Define the primary key.
    string += ' PRIMARY KEY (`%s`),' % (prim_key)

    # Loop through the foreign key names, with tuples of the form:
    # (constraint name, foreign key name, target table, target key)
    for name, tup in constraint_dict.items():
        if tup[3]:
            string += ' CONSTRAINT `%s` FOREIGN KEY (`%s`) REFERENCES `%s` (`%s`) ON DELETE CASCADE' % (name, tup[0], tup[1], tup[2])
        else:
            string += ' CONSTRAINT `%s` FOREIGN KEY (`%s`) REFERENCES `%s` (`%s`)' % (name, tup[0], tup[1], tup[2])

    # Adds the engine type at the end.
    string += ' ) ENGINE = InnoDB'
    return string


def insert_table(metadata, table_name, engine, df):
    business = Table(table_name, metadata)
    connection = engine.connect()
    ins = business.insert()
    records_dict = df.to_dict('records')
    connection.execute(ins, records_dict)

