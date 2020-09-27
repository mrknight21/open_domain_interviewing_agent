#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
from csv import DictReader
from parlai.utils.io import PathManager

RESOURCES = {
    "train": os.path.join("NewsQA", "split_data", "train.csv"),
    "valid": os.path.join("NewsQA", "split_data", "dev.csv"),
    "test": os.path.join("NewsQA", "split_data", "test.csv")
}

OUTPUT_FORMAT = (
    'text:{question}\tanswer_starts:{start}\tlabels:{labels}'
)

def _parse_answers(story, tokens_range_str):
    starts = []
    labels = []
    tokenized_story = story.split()
    tokens_ranges = tokens_range_str.split(',')
    for each in tokens_ranges:
        start_end_ind = each.split(':')
        start_ind, end_ind = start_end_ind[0], start_end_ind[1]
        starts.append(str(start_ind))
        text = tokenized_story[int(start_ind):int(end_ind)].join(' ')
        labels.append(text.replace('|', ' __PIPE__ '))
    return '|'.join(starts), '|'.join(labels)


def _handle_paragraph(row):
    output = []
    story = row['story_text'].replace('\n', '\\n')
    question = row['question']
    question_txt = story + '\\n' + question
    starts, labels = _parse_answers(story, row['answer_token_ranges'])
    output.append(
        OUTPUT_FORMAT.format(
            question=question_txt,
            start=starts,
            labels=labels,
        )
    )
    output.append('\t\tepisode_done:True\n')
    return ''.join(output)


def make_parlai_format(outpath, dtype, data):
    print('building parlai:' + dtype)
    with PathManager.open(os.path.join(outpath, dtype + '.txt'), 'w') as fout:
        for row in data:
            fout.write(_handle_paragraph(row))


def build(opt):
    dpath = os.path.join(opt['datapath'], 'NewsQA', 'single_turn')

    if not build_data.built(dpath):
        print('[building data: ' + dpath + ']')
        build_data.make_dir(dpath)
        for dtype, dtpath in RESOURCES.items():
            data_path = os.path.join(opt['datapath'], dtpath)
            with open(data_path, 'r') as data:
                csv_data = DictReader(data)
                make_parlai_format(dpath, dtype, csv_data)

        # Mark the data as built.
        build_data.mark_done(dpath)