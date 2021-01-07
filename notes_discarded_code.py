#%% Code for creating db and table directly via mysqlconnector. In practice,
# sqlalchemy seems like the more robust option.
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

#%% Table creation from within python.

DB_NAME = 'test_db'

TABLES = {}

# Example of a table that can be imitated with my table string output function.
# Again in practice, sqlalchemy is a much better tool.
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


#%% Test list comprehensions
attribute = 'BusinessAcceptsCreditCards'
attribute_list_noempty = ['' if v is None else v for v in attribute_list]
test_output = [x[attribute] if attribute in x else False for x in attribute_list_noempty]


#%% Actually a list of dicts, each dict is a row of the dataframe where the keys are the column names, values are row values.
# In general, this approach wound up being more convoluted than just writing an entire df to the db with
# df.to_sql()

test_dict = sub_bus_frame.to_dict('records')

#%% Worked. The default columns are written to the db (but not the columns that need to be distributed).
connection.execute(ins, test_dict)

# This metadata object collects the schema from the existing db.
metadata = MetaData()
metadata.reflect(bind=engine)

# Confirmation of schema
tables_dict = metadata.tables

#%%

business_table = metadata.tables['business']
sub_bus_frame = df_businesses.filter(['latitude', 'longitude']).copy()
update_dict = sub_bus_frame.to_dict('records')

#%% Update the longitude/latitude to accurate float values
connection = engine.connect()
upd = (
    update(business_table)
)

connection.execute(upd, update_dict)
