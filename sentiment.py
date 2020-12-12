import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import utils
import gc
import numpy as np
import pickle
#%% Load reviews
path_review = "data\yelp_archive\yelp_academic_dataset_review.json"
df_review = utils.read_json(path_review)

#%% Declare sentiment analysis object and test it on a review.

sid = SentimentIntensityAnalyzer()
test_output = sid.polarity_scores(df_review['text'][0])
print(df_review['text'][0])
print(test_output)

#%% Isolate column and delete df to memory purposes.
df_review_text = pd.DataFrame()
df_review_text['text'] = df_review['text']
del df_review
gc.collect()
Out[10]: 15
df_review=pd.DataFrame()

#%% Split up the dataframe and clean up for memory purposes.

# Chunk size
n = 100000
list_df = np.array_split(df_review_text, n)
del df_review_text
gc.collect()
Out[10]: 15
df_review_text=pd.DataFrame()

#%% Iterate through the sub frames collecting intensity scores.

for frame in list_df:
    frame['vader_scores'] = frame['text'].apply(lambda text: sid.polarity_scores(text))

#%% Merge the list of dataframes and save

df_review_sentiment = pd.concat(list_df)

with open("sentiments.pickle", 'wb') as f:
    pickle.dump(df_review_sentiment, f)