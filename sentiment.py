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


#%% Next steps: set up the model, do an embedding layer with glove embeddings. We'll see how nasty it is.