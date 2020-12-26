### Limited sentiment analysis using review stars as labels to examine
### conformity with the vader lexicon.

import nltk
import pandas as pd
import numpy as np
import utils
import pickle
from importlib import reload
from sklearn.model_selection import train_test_split, GridSearchCV
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
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import regularizers
import timeit



#%% Import the review data, keep 20k for study and pickle

path_review = "data\yelp_archive\yelp_academic_dataset_review.json"
df_review = utils.read_json(path_review)

#%%
df_sentiment_data = df_review[['stars', 'text']][0:20000]

# Drop neutral reviews, since we'll be doing coarse analysis.
df_sentiment_binary = utils.drop_neutral(df_sentiment_data)

with open("review_subset.pickle", 'wb') as f:
    pickle.dump(df_sentiment_binary, f)

#%% Load the review subset

df_sentiment_binary = utils.load_stuff("review_subset.pickle")

#%% Convert the tokenized review to indexed vectors, then convert
# this list of lists to a padded numpy array, where each row is a review. Note that
# we have truncated/padded to a max length of 200 words, favoring the end of reviews
# because this is where more impactful language is likely to be found. The vast majority
# of reviews fit in this length anyway, and long pos/neg reviews are likely to be highly
# redundant in regard to sentiment. 

max_len = 200
stop_set = set(stopwords.words('english'))
review_matrix, mat_sentiment_binary, Vocab = utils.text_to_vectors(df_sentiment_binary, stop_set, 200, Vocab = None)
vocab_size = len(Vocab)
#%% Train/validate/test sets

x_train, x_test, y_train, y_test = (
    train_test_split(review_matrix, mat_sentiment_binary[:, 0], test_size=0.2, random_state=1)
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

#%% Create a Model with a convolutional layer before the LSTM layer.

model = KerasClassifier(build_fn=utils.create_cnn_model, verbose=1)

#%% Hyperparameter tuning vars. Note that gridsearch is being allowed
# to default to k=5 fold cross validation.

# param_results collects the results of various testing. Not that he_normal is named
# for the author, and it's kaiming.
param_results = {'batch_size':20, 'epochs':10, 'activation':'relu', 'optimizer':'adam', 'init_mode':'he_normal', 'units':128, 'rate':0.5}
epochs = [5]
activity_regularizer = ["l2"]
param_dict = dict(activity_regularizer=activity_regularizer, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_dict, n_jobs=1, cv=3)

#%% Perform grid search.

grid_result = grid.fit(x_train, y_train)

# Summary
print("Best Result was: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
params = grid_result.cv_results_['params']
#%% Fresh model for fine-tuning and early stopping

model = utils.create_cnn_model(init_mode='he_uniform', activity_regularizer = regularizers.l2(1e-4))

#%% Load CNN to LSTM model
model_best = load_model('C:\\Projects\yelp_analysis\\best_model.h5', compile=False)

#%% Train the model

# Early stopping callback and model checkpoint, which saves the best model dynamically.
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
callback_list = [es, mc]

training_hist = model.fit(x_train, y_train, batch_size=20, epochs=20, verbose=1, validation_split=0.2, callbacks=callback_list)

result = model.evaluate(x_test, y_test, verbose=1)
print(result)

#%% View history

utils.plot_history(training_hist, 'CNN to LSTM')

#%% Check saved model
model_best.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
result = model_best.evaluate(x_test, y_test, verbose=1)
print(result)
# 90.8% accurate.
#%% Let's compare the vader predictions and my own classifier to the actual
# star scores.

# Set a range of values to test. We'll pull the vader sentiment results and the associated review text together.
a = 100000
b = 200000
df_sentiment_vader = utils.load_df_subset('sentiments.pickle','pickle', a, b, ['vader_scores'])

df_review_subset = utils.load_df_subset(path_review, 'JSON', a, b, ['stars', 'text'])

# Drop neutral values since we are still doing coarse sentiment.
df_review_sub_bin = utils.drop_neutral(df_review_subset)

# Get the tokenized integer form reviews. Note that we MUST pass the Vocab dict used
# to train the original model since the alternative would have been to build the Vocab
# based on the entire dataset.
review_matrix, mat_sentiment_binary, Vocab = utils.text_to_vectors(df_review_sub_bin, stop_set, 200, Vocab=Vocab)

#Load model to evaluate and get predictions.
model_final = load_model('C:\\Projects\yelp_analysis\\best_model.h5', compile=True)
predictions = model_final.predict_classes(review_matrix)

# Make a dataframe of relevant results
df_sentiment_results = pd.DataFrame()

# Get coarse values from the vader sentiment and re-index to match the a,b range.
df_sentiment_results['vader_binary'] = [x['compound'] > 0.5 for x in df_sentiment_vader['vader_scores']]
df_sentiment_results['new_index'] = range(a,b)
df_sentiment_results.set_index('new_index', inplace=True)

# Inner join with the dataframe returned from the drop_neutral function. This has the indexes still, and
# so it can be used with the inner join, and the adjusted indexes of the vader sentiments, to keep the appropriate
# values.
df_sentiment_results = pd.merge(df_sentiment_results, df_review_sub_bin, left_index=True, right_index=True)

# Add predictions and convert vader values to integer booleans, re-order.
df_sentiment_results['predictions'] = predictions
df_sentiment_results['vader_binary'] = [x*1 for x in df_sentiment_results['vader_binary']]
df_sentiment_results = df_sentiment_results.reindex(columns = ['text', 'stars', 'vader_binary', 'predictions'])