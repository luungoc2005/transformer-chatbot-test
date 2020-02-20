import torch
import json
import math

from os import path, listdir
from collections import OrderedDict
from array import array

from tqdm import tqdm

get_temp_path = lambda split_name: f'./data/corpus_{split_name}.pt'
DATA_PATH = './data/reddit/'

splits_ratios = OrderedDict()
splits_ratios['train'] = .7
splits_ratios['val'] = .2
splits_ratios['test'] = .1

class RedditCorpus(object):

    def __init__(self, folder_path=DATA_PATH, split_name='train'):
        self.folder_path = folder_path
        self.files_list = [
            path.join(self.folder_path, item)
            for item in sorted(listdir(self.folder_path))
            if item.endswith('.json')
        ]
        
        assert split_name in splits_ratios.keys()

        start_idx = 0
        end_idx = 0

        for key, value in splits_ratios.items():
            if key != split_name:
                start_idx += math.floor(len(self.files_list) * value)
            else:
                end_idx = start_idx + math.floor(len(self.files_list) * value)
                break

        self.files_list = self.files_list[start_idx:end_idx]

        if split_name == 'train':
            self.files_list.append(path.join(DATA_PATH, '../movies/movies_batch_0.json'))

        temp_file = get_temp_path(split_name)
        # print(self.files_list)
        if not path.exists(temp_file):
            from utils import get_default_tokenizer
            tokenizer = get_default_tokenizer()

            input_ids = array('I')

            p0_token = tokenizer.token_to_id('[P0]')
            p1_token = tokenizer.token_to_id('[P1]')
            sep_token = tokenizer.token_to_id('[SEP]')
            doc_sep_token = tokenizer.token_to_id('[DOC_SEP]')

            for file_name in tqdm(self.files_list):
                with open(file_name, 'r') as input_file:
                    file_content = json.load(input_file)

                for example in tqdm(file_content, leave=False):
                    context_length = len(example['context'])
                    for ix, line in enumerate(example['context']):
                        
                        if (context_length % 2 == 0):
                            input_ids.append(p0_token if ix % 2 == 0 else p1_token)
                        else:
                            input_ids.append(p1_token if ix % 2 == 0 else p0_token)
                        
                        input_ids.extend(tokenizer.encode(line).ids + [sep_token])

                    input_ids.extend([p0_token] + tokenizer.encode(example['response']).ids + [sep_token])

                    input_ids.append(doc_sep_token)
            
            self.input_tensor = torch.LongTensor(input_ids)
            del input_ids

            torch.save(self.input_tensor, temp_file)

        else:
            self.input_tensor = torch.load(temp_file)

    def batchify(self, batch_size):
        data = self.input_tensor
        # Divide the dataset into bsz parts.
        nbatch = data.size(0) // batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * batch_size)
        # Evenly divide the data across the bsz batches.
        data = data.view(batch_size, -1).t().contiguous()
        return data.to('cpu')

if __name__ == "__main__":
    print('train')
    test_train_dataset = RedditCorpus(DATA_PATH, 'train')

    print('val')
    test_val_dataset = RedditCorpus(DATA_PATH, 'val')

    print('test')
    test_test_dataset = RedditCorpus(DATA_PATH, 'test')

    