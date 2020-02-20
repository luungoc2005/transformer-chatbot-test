
import argparse
import hashlib
import json
import logging
import os
import re
import uuid
import random

from collections import defaultdict, namedtuple
from functools import partial
from os import path
from tqdm import tqdm

from pymongo import MongoClient

# Represent a reddit comment.
Comment = namedtuple(
    "Comment",
    [
        "id",
        "thread_id",
        "parent_id",
        "body",
        "body_is_trimmed",
        "author",
        "subreddit",
    ]
)


def normalise_comment(comment, max_length):
    """Create a _Comment object from a row in the BigQuery table."""
    return Comment(
        id=comment['id'],
        thread_id=_normalise_id(comment['link_id']),
        parent_id=_normalise_id(comment['parent_id']),
        body=trim(comment['body'], max_length),
        body_is_trimmed=len(comment['body']) > max_length,
        author=comment['author'],
        subreddit=comment['subreddit'],
    )


def _normalise_id(raw_id):
    """Reddit IDs start with t1_, t2_, etc. which need to be stripped."""
    return re.sub("^t[0-9]_", "", raw_id)


def trim(text, max_length):
    """Trims text to be at most `max_length`, without splitting apart words."""
    if len(text) <= max_length:
        return text

    text = text[:max_length + 1]

    # Trim until the last two characters are the boundary between an
    # alphanumeric character, and a non-alphanumeric character.
    while len(text) > 1 and (text[-1].isalnum() == text[-2].isalnum()):
        text = text[:-1]

    return text[:-1]


def _should_skip(comment, min_length):
    if comment.body_is_trimmed:
        return True
    if comment.body in {"[deleted]", "[removed]"}:
        return True
    if len(comment.body) < min_length:
        return True
    return False


def create_examples(thread, parent_depth, min_length, format='JSON'):

    id_to_comment = {comment.id: comment for comment in list(thread)}

    for linear_path in linear_paths(id_to_comment, parent_depth):
        response = id_to_comment[linear_path[-1]]
        context = id_to_comment[linear_path[-2]]  # guaranteed to exist.

        if (_should_skip(response, min_length)
                or _should_skip(context, min_length)):
            continue

        example = {}
        example['subreddit'] = response.subreddit
        example['thread_id'] = response.thread_id
        example['context_author'] = context.author
        example['response_author'] = response.author
        example['context'] = context.body
        example['response'] = response.body

        for i in range(parent_depth - 1):
            # Extra contexts start at index -3.
            index = -3 - i
            try:
                context_i = linear_path[index]
            except IndexError:
                break

            example['context/{}'.format(i)] = id_to_comment[context_i].body

        yield example


def linear_paths(id_to_comment, parent_depth):
    """Gets all linear paths of comments and replies from the thread.
    Each linear path is guaranteed to have at least two comments in it.
    """
    paths = []
    seen_ids = set()
    id_to_children = defaultdict(list)
    for comment_id, comment in id_to_comment.items():
        id_to_children[comment.parent_id].append(comment_id)
        if comment.parent_id not in id_to_comment:
            paths.append([comment_id])
            seen_ids.add(comment_id)

    while paths:
        new_paths = []
        for path in paths:
            last_id = path[-1]
            for child_id in id_to_children[last_id]:
                if child_id in seen_ids:
                    # Prevent infinite loops.
                    continue
                seen_ids.add(child_id)
                new_path = path[-parent_depth:] + [child_id]
                new_paths.append(new_path)
                yield new_path
        paths = new_paths

if __name__ == "__main__":
    client = MongoClient('localhost', 27017)

    db = client['reddit_texts']
    all_collection = db['all']

    all_threads = all_collection.aggregate([
        { '$group': { '_id': '$link_id' } }
    ], batchSize=1024)

    pbar = tqdm()
    batch = []
    max_batch_size = 500000 # over 500k
    batch_id = 0
    example_id = 0
    parent_depth = 10

    for thread_obj in all_threads:
        thread_id = thread_obj['_id']
        all_comments = all_collection.find({ 'link_id': thread_id })
        all_comments = [normalise_comment(item, 256) for item in all_comments]

        for example in create_examples(all_comments, parent_depth, 9):
            batch.append(example)
            example_id += 1
            pbar.set_description_str(f'examples: {example_id}, batches: {batch_id}')
            pbar.update()

            if len(batch) > max_batch_size:
                random.shuffle(batch)
                raw_file = open(path.join('./data/reddit/raw', f'batch_{batch_id}.raw.txt'), 'w')
                p_file = open(path.join('./data/reddit', f'batch_{batch_id}.json'), 'w')
                # write batch to file

                p_content = []
                for batch_example in batch:

                    batch_context = []
                    # P0: response_author
                    # P1: context_author
                    for ix in range(parent_depth, 0, -1):
                        context_line = batch_example.get(f'context/{ix}')
                        if context_line is None:
                            continue
                        
                        batch_context.append(context_line.strip())
                        raw_file.write(context_line.strip() + ' ')
                        # p_file.write(f' [P{ix % 2}] ' + context_line.strip() + ' [SEP] ')

                    raw_file.write(batch_example["context"].strip() + ' ')
                    raw_file.write(batch_example["response"].strip() + ' ')

                    batch_context.append(batch_example["context"].strip())

                    p_content.append({
                        'context': batch_context,
                        'response': batch_example["response"].strip()
                    })
                    # p_file.write(f' [P0] {batch_example["context"].strip()} [SEP] ')
                    # p_file.write(f' [P1] {batch_example["response"].strip()} [SEP] ')
                    # p_file.write(f' [DOC_SEP] ')
                    # print(batch_example)

                json.dump(p_content, p_file)
                p_file.flush()
                p_file.close()

                raw_file.flush()
                raw_file.close()

                # cleanup
                batch = []

                batch_id += 1

    exit()