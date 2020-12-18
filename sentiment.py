### Limited sentiment analysis using review stars as labels to examine
### conformity with the vader lexicon.

import nltk
import pandas as pd
import numpy as np
import utils
import pickle
from importlib import reload
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download("stopwords")
from keras.preprocessing.sequence import pad_sequences
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM, Conv1D, TimeDistributed, MaxPooling1D
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint

#%% Import the review data, keep 20k for study and pickle

path_review = "data\yelp_archive\yelp_academic_dataset_review.json"
df_review = utils.read_json(path_review)

df_sentiment_data = df_review[['stars', 'text']][0:20000]

# Drop neutral reviews, since we'll be doing coarse analysis.
df_sentiment_binary = df_sentiment_data.drop(df_sentiment_data[df_sentiment_data['stars'] == 3].index)

with open("review_subset.pickle", 'wb') as f:
    pickle.dump(df_sentiment_binary, f)

#%% Load the review subset

df_sentiment_binary = utils.load_stuff("review_subset.pickle")

#%% Switch to numpy arrays
mat_sentiment_binary = df_sentiment_binary.to_numpy()
mat_sentiment_binary[:,0] = (mat_sentiment_binary[:,0] > 3) * 1

#%% Tokenize and drop stop words from corpus
stop_set = set(stopwords.words('english'))
sent_list = list(mat_sentiment_binary[:,1])
reviews_cleaned = [[word for word in word_tokenize(sent) if word not in stop_set] for sent in sent_list]
#%% Build the vocabulary

# Include special tokens
# started with pad, end of line and unk tokens
Vocab = {'__PAD__': 0, '__</e>__': 1, '__UNK__': 2}

# Note that we build vocab using training data
for review in reviews_cleaned:
    for word in review:
        if word not in Vocab:
            Vocab[word] = len(Vocab)

print("Total words in vocab are", len(Vocab))
vocab_size = len(Vocab)

#%% Convert the tokenized review to indexed vectors, then convert
# this list of lists to a padded numpy array, where each row is a review. Note that
# we have truncated/padded to a max length of 200 words, favoring the end of reviews
# because this is where more impactful language is likely to be found. The vast majority
# of reviews fit in this length anyway, and long pos/neg reviews are likely to be highly
# redundant in regard to sentiment. 
max_len = 200
word_list_vec = [utils.review_to_vector(list, Vocab, '__UNK__') for list in reviews_cleaned]
review_matrix = pad_sequences(word_list_vec, value=0, maxlen=max_len)
#%% Train/validate/test sets

x_train, x_test, y_train, y_test = (
    train_test_split(review_matrix, mat_sentiment_binary[:, 0], test_size= .2, random_state=1)
)



#%%
sent_vector = mat_sentiment_binary[:,0]
sns.set_theme(style="darkgrid")
axs = sns.countplot(x = sent_vector)
plt.show()

#%% Import glove embeddings
embeddings_dict = dict()
glove_file = open('C:\\Users\zennsunni\Documents\Glove Embeddings\glove.6B\glove.6B.100d.txt', encoding="utf8")
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_array = np.asarray(records[1:], dtype='float32')
    embeddings_dict[word] = vector_array
glove_file.close()

#%% Create embedding matrix

embedding_mat = np.zeros((vocab_size, 100))
for word, index in Vocab.items():
    vec = embeddings_dict.get(word)
    if vec is not None:
        embedding_mat[index] = vec
#%% Save the embeddings matrix
with open("embedding_matrix.pickle", 'wb') as f:
    pickle.dump(embedding_mat, f)
#%% Load the embedding matrix
embedding_mat = utils.load_stuff("embedding_matrix.pickle")

#%% Create a LTSM Model.
model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_mat], input_length=200 , trainable=False)
model.add(embedding_layer)
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

#%% Save the model, as compilation time is high
model.save('C:\Projects\yelp_analysis\keras_sentiment_model', overwrite=True)
#%% Load model
model = load_model('C:\Projects\yelp_analysis\keras_sentiment_model')
#%% Create a Model with a convolutional layer before the LSTM layer.
model_cnn_lstm = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_mat], input_length=200 , trainable=False)
model_cnn_lstm.add(embedding_layer)
model_cnn_lstm.add(Conv1D(filters=32, kernel_size=4, padding='same', activation='relu'))
model_cnn_lstm.add(MaxPooling1D(pool_size=2))
model_cnn_lstm.add(LSTM(128))
model_cnn_lstm.add(Dense(128, activation='relu'))
model_cnn_lstm.add(Dropout(.5))
model_cnn_lstm.add(Dense(1, activation='sigmoid'))
model_cnn_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model_cnn_lstm.summary())

#%% Save the CNN/LSTM Model
model_cnn_lstm.save('C:\Projects\yelp_analysis\keras_sentiment_model_cnn', overwrite=True)
#%% Load CNN to LSTM model
model_cnn_lstm = load_model('C:\Projects\yelp_analysis\keras_sentiment_model_cnn', compile=False)

#%% Train the model

# Early stopping callback and model checkpoint, which saves the best model dynamically.
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
callback_list = [es, mc]

training_hist = model_cnn_lstm.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1, validation_split=0.2, callbacks=callback_list)

result = model_cnn_lstm.evaluate(x_test, y_test, verbose=1)
print(result)

#%% View history

utils.plot_history(training_hist, 'CNN to LSTM')