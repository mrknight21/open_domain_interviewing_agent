
from parlai.tasks.squad2.agents import IndexTeacher
from parlai_internal.utilities import util
from transformers.data.processors.squad import SquadExample

import copy

NO_ANSWER_REPLY = "[CLS]"

class DefaultTeacher(IndexTeacher):

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        super().__init__(opt, shared)

    def get(self, episode_idx, entry_idx=None):
        is_training = self.datatype == "train"
        article_idx, paragraph_idx, qa_idx = self.examples[episode_idx]
        article = self.squad[article_idx]
        paragraph = article['paragraphs'][paragraph_idx]
        context_text = paragraph["context"]
        qa = paragraph["qas"][qa_idx]
        qas_id = qa["id"]
        question_text = qa["question"]
        start_position_character = None
        answer_text = None
        answers = []
        is_impossible = qa.get("is_impossible", False)
        if not is_impossible:
            answer = qa["answers"][0]
            answer_text = answer["text"]
            start_position_character = answer["answer_start"]
            answers = [qa['text'] for qa in qa["answers"]]
        else:
            answers = [NO_ANSWER_REPLY]

        squad_example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title="unknown title",
                        is_impossible=is_impossible,
                        answers=answers,
                    )

        action = {
            'id': 'squad',
            'qas_id': qas_id,
            'context': context_text,
            'labels': answers,
            'squad_example': squad_example,
            'single_label_text': answer_text,
            'episode_done': True,
            'is_impossible': is_impossible,
            'no_answer_reply': NO_ANSWER_REPLY
        }
        return action


