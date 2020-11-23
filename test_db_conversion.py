import pandas as pd
import json

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
