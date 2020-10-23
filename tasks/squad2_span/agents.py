
from parlai.tasks.squad2.agents import IndexTeacher

import copy

NO_ANSWER_REPLY = "I don't know"

class DefaultTeacher(IndexTeacher):

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        super().__init__(opt, shared)


    def get(self, episode_idx, entry_idx=None):
        article_idx, paragraph_idx, qa_idx = self.examples[episode_idx]
        article = self.squad[article_idx]
        paragraph = article['paragraphs'][paragraph_idx]
        qa = paragraph['qas'][qa_idx]
        question = qa['question']
        answers = []
        answer_starts = []
        answer_ends = []
        context = paragraph['context']
        if not qa['is_impossible']:
            for a in qa['answers']:
                answers.append(a['text'])
                start_idx, end_idx = self.get_start_end_idx(a['text'], a['answer_start'], context)
                answer_starts.append(start_idx)
                answer_ends.append(end_idx)
        else:
            answers = [self.opt['impossible_answer_string']]

        plausible = qa.get("plausible_answers", [])

        action = {
            'id': 'squad',
            'text': context + ' [SEP] ' + question,
            'context': context,
            'question': question,
            'labels': answers,
            'plausible_answers': plausible,
            'episode_done': True,
            'answer_starts': answer_starts,
            'answer_ends': answer_ends,
            'is_impossible': qa['is_impossible']
        }
        return action

    def get_start_end_idx(self, gold_text, start_idx, context):
        end_idx = start_idx + len(gold_text)
        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            return start_idx, end_idx
        elif context[start_idx - 1:end_idx - 1] == gold_text:
            start_idx = start_idx - 1
            end_idx = end_idx - 1  # When the gold label is off by one character
        elif context[start_idx - 2:end_idx - 2] == gold_text:
            start_idx = start_idx - 2
            end_idx = end_idx - 2  # When the gold label is off by two characters
        return start_idx, end_idx
