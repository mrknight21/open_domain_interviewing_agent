import torch

from parlai.core.message import Message
from parlai.core.torch_agent import Optional, Batch, Output
from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai_internal.utilities.flow_lstm_util.dictionary_agent import InterviewDictionaryAgent
from parlai_internal.agents.torch_span_agent.torch_span_agent import DialogueHistory
from parlai_internal.utilities.flow_lstm_util.models.seq2seq import Seq2SeqModel

class InterviewerHistory(DialogueHistory):

    def __init__(self, opt, **kwargs):
        self.sep_last_utt = opt.get('sep_last_utt', False)
        super().__init__(opt, **kwargs)
        self.title = None
        self.background = None
        self.section_title = None
        self.history_cache = []

    def reset(self):
        """
        Clear the history.
        """
        self.history_raw_strings = []
        self.history_cache = []
        self.history_dialogues = []
        self.history_strings = []
        self.history_vecs = []
        self.context = None
        self.title = None
        self.background = None
        self.section_title = None

    def _update_cache(self, obs):
        cache = {
            'character_start_end': obs['character_start_end'],
            'yesno': obs['yesno'], 'followup': obs['followup']}
        self.history_cache.append(cache)

    def update_history(self, obs: Message, temp_history: Optional[str] = None):
        """
        Update the history with the given observation.

        :param obs:
            Observation used to update the history.
        :param temp_history:
            Optional temporary string. If it is not None, this string will be
            appended to the end of the history. It will not be in the history
            on the next dialogue turn. Set to None to stop adding to the
            history.
        """
        if "text" in obs and obs["text"] is not None:
            if not self.context and obs.get('context', None):
                    self.context = obs['context']
            if not self.background and obs.get('background', None):
                    self.background = obs['background']
            if not self.title and obs.get('title', None):
                    self.title = obs['title']
            if not self.section_title and obs.get('section_title', None):
                    self.section_title = obs['section_title']
            text = obs['text']
            self._update_raw_strings(text)
            if self.add_person_tokens:
                text = self._add_person_tokens(
                    obs[self.field], self.p1_token, self.add_p1_after_newln
                )
            # update history string
            self._update_strings(text)
            # update history dialogues
            self._update_dialogues(text)
            # update history vecs
            self._update_vecs(text)
            self._update_cache(obs)
        self.temp_history = temp_history



class InterviewerAgent(TorchGeneratorAgent):
    """
    Interviewer agent.
    """

    @staticmethod
    def dictionary_class():
        """
        Return the dictionary class that this agent expects to use.

        Can be overriden if a more complex dictionary is required.
        """
        return InterviewDictionaryAgent

    @classmethod
    def history_class(cls):
        """
        Return the history class that this agent expects to use.

        Can be overriden if a more complex history is required.
        """
        return InterviewerHistory

    def build_model(self):
        """
        Construct the model.
        """
        model = self.load_question_generation_model()
        self.criterion = None
        return model

    def load_question_generation_model(self):

        filename = self.opt['init_model_file']
        print(f"Loading model from '{filename}'...")
        checkpoint = torch.load(filename, lambda storage, loc: storage)
        args = checkpoint['config']
        if self.dict.vocab is not None:
            args['vocab'] = self.dict.vocab
        self.model = Seq2SeqModel(args, use_cuda=self.use_cuda)
        self.model.load_state_dict(checkpoint['model'])