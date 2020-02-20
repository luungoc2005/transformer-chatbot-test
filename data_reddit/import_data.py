from pymongo import MongoClient
from os import path, listdir
import bz2
import json

data_path = '/media/luungoc2005/Data/Projects/Data/2015_reddit_comments_corpus/reddit_data'

client = MongoClient('localhost', 27017)

db = client['reddit_texts']
all_collection = db['all']

sub_dirs = [
    path.join(data_path, item) for item in listdir(data_path) 
    if path.isdir(path.join(data_path, item))
]


# take the first 3 years?
sub_dirs = sub_dirs[:3]

for sub_dir in sub_dirs:
    print(sub_dir)
    sub_files = [
        path.join(sub_dir, item) for item in listdir(sub_dir) 
        if item.endswith('.bz2')
    ]
    for sub_file in sub_files:
        print('Processing ' + sub_file)
        with bz2.open(sub_file, 'r') as f:
            # file contents:
            contents = [json.loads(line) for line in f.readlines()]
            all_collection.insert_many(contents, ordered=False)