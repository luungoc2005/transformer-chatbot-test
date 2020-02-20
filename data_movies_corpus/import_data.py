import pandas as pd
from os import path, listdir
from tqdm import tqdm
import zipfile
import json

DATA_PATH = './data/cornell_movie_dialogs_corpus.zip'
SAVE_PATH = './data/movies'
DELIMITER = '+++$+++'
ENCODING = 'iso-8859-1'

dialogues = []
content = []
convo_count = 0
file_count = 0

if path.exists(DATA_PATH):
    with zipfile.ZipFile(DATA_PATH, 'r') as source_zip:
        lines_map = {}

        with source_zip.open('cornell movie-dialogs corpus/movie_lines.txt', 'r') as lines_file:
            lines_list = lines_file.readlines()
            for lines_row in lines_list:
                lines_row = lines_row.decode(ENCODING).split(DELIMITER)
                lines_map[lines_row[0].strip()] = lines_row[-1].strip()

        with source_zip.open('cornell movie-dialogs corpus/movie_conversations.txt', 'r') as convo_file:
            convo_list = convo_file.readlines()
            for convo_row in convo_list:
                convo_row = convo_row.decode(ENCODING).split(DELIMITER)
                convo_lines = json.loads(convo_row[-1].strip().replace("'", '"'))
                convo_lines = [lines_map[item] for item in convo_lines if item in lines_map]
                dialogues.append(convo_lines)


    def flush_to_file():
        global content, file_count, convo_count
        
        file_name = path.join(SAVE_PATH, f'movies_batch_{file_count}.json')
        with open(file_name, 'w') as tmp_file:
            json.dump(content, tmp_file)

        content = []
        convo_count = 0
        file_count += 1

    for convo in tqdm(dialogues):
        if len(convo) >= 2:
            convo_count += 1
            content.append({
                'context': convo[:-1],
                'response': convo[-1]
            })

        if convo_count > 500000:
            flush_to_file()

    flush_to_file()