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
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding

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

x_train, x_val, y_train, y_val = (
    train_test_split(x_train, y_train, test_size=0.25, random_state=1)
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

#%% Create the model. We're using an LTSM layer at the core of this model,
# since LTSM networks perform very well for sequence data.
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
#%% Train the model

training_hist = model.fit(x_train, y_train, batch_size=128, epochs=1, verbose=1, validation_split=0.2)

result = model.evaluate(x_test, y_test, verbose=1)