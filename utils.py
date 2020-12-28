import mysql.connector
from mysql.connector import errorcode
import json
import pandas as pd
from sqlalchemy import create_engine
import pymysql
from sqlalchemy import MetaData, Column, insert, Table
import pickle
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM, Conv1D, TimeDistributed, MaxPooling1D, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from tensorflow.keras import regularizers
from keras.preprocessing.sequence import pad_sequences


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

def load_stuff(path):
    '''
    Loads a file
    Parameters
    ----------
    path: str
        local or full path of file

    Returns
    -------
    '''
    with open(path, 'rb') as f:
        file = pickle.load(f)
    return file

# Converts reviews to vectors of indexed numbers.
def review_to_vector(review, vocab_dict, unk_key):
    '''
    Input:
        review - The review as a tokenized list
        vocab_dict - The words dictionary
    Output:
        vector_list - a python list of integer values associated with vocab words

    '''
    # Convert the tokenized review into a list of integers indexed in the vocab,
    # using the unknown value if the word isn't present.
    unkn_id = vocab_dict[unk_key]
    vector_list = [vocab_dict[word] if word in vocab_dict else unkn_id for word in review]

    return vector_list

def plot_history(hist_object, model_type):
    plt.plot(hist_object.history['acc'])
    plt.plot(hist_object.history['val_acc'])
    plt.title('Early Stopping Accuracy for %s Architecture' %(model_type))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# Model creation function for use with sklearn wrapper.
def create_cnn_model(filters=32, kernel_size=4, activation='relu',
                     dense1_activ = 'relu', rate = .5,
                     units=128, optimizer='adam', init_mode='uniform', activity_regularizer=None):
    '''
    Creates a model of the (informal) architecture:
     Embedding -> CNN -> Maxpool -> LSTM -> Dense -> Dropout -> Dense
    :return:
        Keras Sequential Model
    '''
    embedding_mat = load_stuff("embedding_matrix.pickle")
    model = Sequential()
    embedding_layer = Embedding(49433, 100, weights=[embedding_mat], input_length=200, trainable=False)
    model.add(embedding_layer)
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation=activation))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(128))
    model.add(Dense(units=units, activation='relu', kernel_initializer=init_mode, activity_regularizer=activity_regularizer))
    model.add(Dropout(rate=rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])
    return model


def load_df_subset(path, type, a, b, col_list):
    '''
    Returns a dataframe with a selected range of rows from a JSON file.
    :param path: str
        Path to a pickled dataframe or JSON file..
    :param a: int
        Starting index.
    :param b: int
        Ending index.
    :param col_list: list
        List of column headings (strings).
    :return: DataFrame
    '''
    if type == 'pickle':
        df = load_stuff(path)
    elif type == 'JSON':
        df = read_json(path)
    else:
        raise ValueError('Function only loads files of type "pickle" or "JSON".')

    df = df[a:b]
    df_subset = df[[c for c in df.columns if c in col_list]]
    return df_subset

def text_to_vectors(df, stop_set, max_len, Vocab = None):
    '''
    Takes in a dataframe of two columns, star ratings for a review and review text, and returns a numpy
    matrix of the text reviews as rows of integer tokens, a numpy matrix of binary reviews and review text,
    and a vocabulary dictionary of the tokens.
    :param df: DataFrame
        A dataframe of the stars and review text.
    :param stop_set: set
        The set of stop words.
    :param max_len: int
        Max tokenized review length.
    :return: ndarray, ndarray, dict
    '''
    mat_sentiment_binary = df.to_numpy()
    mat_sentiment_binary[:, 0] = (mat_sentiment_binary[:, 0] > 3) * 1
    sent_list = list(mat_sentiment_binary[:, 1])
    reviews_cleaned = [[word for word in word_tokenize(sent) if word not in stop_set] for sent in sent_list]
    if Vocab == None:
        Vocab = {'__PAD__': 0, '__</e>__': 1, '__UNK__': 2}

        # Note that we build vocab using training data
        for review in reviews_cleaned:
            for word in review:
                if word not in Vocab:
                    Vocab[word] = len(Vocab)

    print("Total words in vocab are", len(Vocab))
    word_list_vec = [review_to_vector(list, Vocab, '__UNK__') for list in reviews_cleaned]
    review_matrix = pad_sequences(word_list_vec, value=0, maxlen=max_len)
    return review_matrix, mat_sentiment_binary, Vocab

def drop_neutral(df):
    '''
    Drops 3 star reviews from the dataframe.
    :param df: DataFrame
    :return: DataFrame
    '''
    df_binary = df.drop(df[df['stars'] == 3].index)
    return df_binary

def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)