#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
import json
from parlai.core.build_data import DownloadableFile
from parlai.utils.io import PathManager

RESOURCES = [
    DownloadableFile(
        'https://s3.amazonaws.com/my89public/quac/train_v0.2.json',
        'train_v0.2.json',
        'ff5cca5a2e4b4d1cb5b5ced68b9fce88394ef6d93117426d6d4baafbcc05c56a',
        zipped=False,
    ),
    DownloadableFile(
        'https://s3.amazonaws.com/my89public/quac/val_v0.2.json',
        'val_v0.2.json',
        '09e622916280ba04c9352acb1bc5bbe80f11a2598f6f34e934c51d9e6570f378',
        zipped=False,
    ),
]

VERSION = '0.2'

SHOULD = '__SHOULD__'
MAYBE = '__MAYBE__'
SHOULD_NOT = '__SHOULDNOT__'

YES = '__YES__'
NO = '__NO__'
NEITHER = '__NEITHER__'

MAP_CONTINUATION = {'m': MAYBE, 'f': SHOULD, 'n': SHOULD_NOT}
MAP_AFFIRMATION = {'y': YES, 'n': NO, 'x': NEITHER}

OUTPUT_FORMAT = (
    'text:{answer}\tfollowup:{continuation}\tyesno:'
    '{affirmation}\tanswer_starts:{start}\tlabels:{labels}\t'
    'context:{context}\ttitle:{title}\tsection_title:{section_title}\t'
    'background:{background}\tturn_id:{turn_id}'
)


def _parse_answers(q_a):
    starts = []
    labels = []
    for each in q_a['answers']:
        starts.append(str(each['answer_start']))
        labels.append(each['text'].replace('|', ' __PIPE__ '))
    return '|'.join(starts), '|'.join(labels)


def _handle_paragraph(each, background_info=None):
    output = []
    answers = []
    continuations = []
    affirmations = []
    starts = []
    for idx, q_a in enumerate(each['qas']):
        question_txt = q_a['question']
        title = ""
        section_title = ""
        background = ""
        context = ""
        if idx == 0:
            context = each['context'].replace('\n', '\\n')
            if background_info:
                title = background_info.get('title', "")
                section_title = background_info.get('section_title', "")
                background = background_info.get('background', "")
            output.append(
                OUTPUT_FORMAT.format(
                    answer="",
                    continuation="",
                    affirmation="",
                    start="",
                    labels=question_txt,
                    context=context,
                    title=title,
                    section_title=section_title,
                    background=background,
                    turn_id=str(idx)
                )
            )
        else:
            output.append(
                OUTPUT_FORMAT.format(
                    answer=answers[-1],
                    continuation=continuations[-1],
                    affirmation=affirmations[-1],
                    start=starts[-1],
                    labels=question_txt,
                    context="",
                    title="",
                    section_title="",
                    background="",
                    turn_id=str(idx)
                )
            )
        start, labels = _parse_answers(q_a)
        starts.append(start)
        answers.append(labels)
        continuations.append(MAP_CONTINUATION.get(q_a['followup']))
        affirmations.append(MAP_AFFIRMATION.get(q_a['yesno']))
        output.append('\n')
    output.append(
        OUTPUT_FORMAT.format(
            answer=answers[-1],
            continuation=continuations[-1],
            affirmation=affirmations[-1],
            start=starts[-1],
            labels="",
            context="",
            title="",
            section_title="",
            background="",
            turn_id=str(idx+1)
        )
    )
    output.append('\t\tepisode_done:True\n')
    return ''.join(output)


def make_parlai_format(outpath, dtype, data):
    print('building parlai:' + dtype)
    with PathManager.open(os.path.join(outpath, dtype + '.txt'), 'w') as fout:
        for line in data:
            background_info = {'title': line['title'],
                               'section_title': line['section_title'],
                               'background': line['background']}
            for each in line['paragraphs']:
                fout.write(_handle_paragraph(each, background_info))


def build(opt):
    dpath = os.path.join(opt['datapath'], 'QuACQuestions')
    version = VERSION

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        with PathManager.open(os.path.join(dpath, RESOURCES[0].file_name)) as f:
            data = json.load(f)['data']
            make_parlai_format(dpath, 'train', data)

        with PathManager.open(os.path.join(dpath, RESOURCES[1].file_name)) as f:
            data = json.load(f)['data']
            make_parlai_format(dpath, 'valid', data)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
