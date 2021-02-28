import os
import copy

from transformers.data.processors.squad import SquadExample
from parlai.core.teachers import ParlAIDialogTeacher
from parlai.utils.misc import warn_once
from .build import build


NO_ANSWER_REPLY = "CANNOTANSWER"


def _path(opt):
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    if dt == 'test':
        warn_once('WARNING: Test set not included. Setting datatype to valid.')
        dt = 'valid'
    return os.path.join(opt['datapath'], 'QuACFull', dt + '.txt')

class DefaultTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['parlaidialogteacher_datafile'] = _path(opt)
        super().__init__(opt, shared)

    def get(self, episode_idx, entry_idx=None):
        """
        Get a specific example from the dataset.
        """
        ex = self.episodes[episode_idx][entry_idx]
        is_training = self.datatype == "train"
        qas_id = str(episode_idx) + "_" + str(entry_idx)
        question_text = ex['text']
        answer_text = ex['labels'][0]
        start_position_character = None
        is_impossible = answer_text == NO_ANSWER_REPLY
        # if not is_impossible:
        answers = ex['labels']
        if not is_training:
            start_position_character = int(ex["answer_starts"].split('|')[0])
        else:
            start_position_character = int(ex["answer_starts"])
        char_start_end = (start_position_character, start_position_character + len(answer_text))
        # else:
        #     answers = [NO_ANSWER_REPLY]
        #     char_start_end = (-1, -1)


        squad_example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=self.episodes[episode_idx][0]['context'],
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=ex['title'],
                        is_impossible=is_impossible,
                        answers=answers,
                    )

        action = {
            'id': 'squad',
            'turn_id': ex['turn_id'],
            'qas_id': qas_id,
            'labels': answers,
            'context': ex['context'],
            'squad_example': squad_example,
            'single_label_text': answer_text,
            'episode_done': ex['episode_done'],
            'is_impossible': is_impossible,
            'followup': ex['followup'],
            'yesno': ex['yesno'],
            'text': question_text,
            'no_answer_reply': NO_ANSWER_REPLY,
            'background': ex['background'],
            'section_title': ex['section_title'],
            'title': ex['title'],
            'character_start_end': char_start_end
        }
        return action


