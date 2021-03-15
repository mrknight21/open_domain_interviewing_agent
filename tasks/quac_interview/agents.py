import os
import copy
from parlai_internal.agents.interviewee.interviewee import IntervieweeAgent
from parlai.core.teachers import ParlAIDialogTeacher
from parlai.utils.misc import warn_once
from .build import build
import torch

NO_ANSWER_REPLY = "CANNOTANSWER"


def _path(opt):
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    if dt == 'test':
        warn_once('WARNING: Test set not included. Setting datatype to valid.')
        dt = 'valid'
    return os.path.join(opt['datapath'], 'QuACQuestions', dt + '.txt')

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
        answer_text = ex['text']
        question_text = ex['labels'][0]
        # is_impossible = answer_text == NO_ANSWER_REPLY
        start = ex["answer_starts"]
        if start:
            if not is_training:
                start_position_character = int(ex["answer_starts"].split('|')[0])
            else:
                start_position_character = int(ex["answer_starts"])
            char_start_end = (start_position_character, start_position_character + len(answer_text))
        else:
            char_start_end = (-1, -1)

        action = {
            'id': 'quac',
            'turn_id': ex['turn_id'],
            'qas_id': qas_id,
            'labels': ex['labels'],
            'context': ex['context'],
            'single_label_text': question_text,
            'episode_done': ex['episode_done'],
            # 'is_impossible': is_impossible,
            'followup': ex['followup'],
            'yesno': ex['yesno'],
            'text': answer_text,
            'no_answer_reply': NO_ANSWER_REPLY,
            'background': ex['background'],
            'section_title': ex['section_title'],
            'title': ex['title'],
            'character_start_end': char_start_end
        }
        return action

class ReinforcementLearningTeacherAgent(DefaultTeacher, IntervieweeAgent):

    def __init__(self, opt, shared=None):
        self.opt = opt
        self.dict = self.build_dictionary()
        self.query_truncate = opt['query_maximum_length']
        self.context_truncate = opt['context_maximum_length']
        self.history_truncate = opt['history_maximum_length']
        # now set up any fields that all instances may need
        self.EMPTY = torch.zeros(0, dtype=torch.long)
        self.NULL_IDX = self.dict[self.dict.null_token]
        self.START_IDX = self.dict[self.dict.start_token]
        self.END_IDX = self.dict[self.dict.end_token]
        self.model = self.build_model()
        self.model.eval()
        self.truncate = self.dict.maxtokens
        self.history = self.build_history()
        super().__init__(opt, shared)


    def get_model_answer(self, history):
        pass

    def get_reward(self, history):
        return 0

    def trim_conversation_lineages(self):
        pass

    def observe(self, observation):
        """
        Process observation for metrics.
        """
        self.metrics.clear_recent()
        if hasattr(self, 'lastY') and self.lastY is not None:
            self.metrics.evaluate_response(observation, self.lastY)
            self.custom_evaluation(self.last_act, self.lastY, observation)
            self.lastY = None
        recent_metrics = self.metrics.report_recent()
        if recent_metrics:
            # for display purposes (display_model), take all accumulated
            # metrics back into the original observation. This is an abuse of
            # Messages being pointers
            if 'metrics' in observation:
                # override agent-level metrics if present
                observation.pop('metrics')
            observation['metrics'] = recent_metrics
        return observation