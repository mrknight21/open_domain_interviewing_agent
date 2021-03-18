from parlai.core.message import Message
from parlai.utils.misc import AttrDict
from parlai.core.torch_agent import History, Optional
from collections import deque
import copy


class DialogueTurn(AttrDict):

    def __init__(self, question_text, answer_text=None, log_prob=None, reward=None, **kwargs,):
        super().__init__(
            question=question_text,
            answer=answer_text,
            log_prob=log_prob,
            reward=reward,
            **kwargs,
        )
        self.complete = False
        self.generated = False
        self.question = question_text
        self.answer = answer_text
        self.log_prob = log_prob
        if log_prob:
            self.generated = True
        if self.question and self.answer:
            self.complete = True


    def update(self, answer_text=None, log_prob=None, reward=None):
        if answer_text:
            self.anwer = answer_text
            self.complete = True
        if log_prob:
            self.log_prob = log_prob
            self.generated = True
        if reward:
            self.reward = reward


class DialogueHistory(History):
    def __init__(self, opt, **kwargs):
        self.sep_last_utt = opt.get('sep_last_utt', False)
        super().__init__(opt, **kwargs)
        self.context = None
        self.dialogues = []

    def reset(self):
        """
        Clear the history.
        """
        self.history_raw_strings = []

        self.history_strings = []
        self.history_vecs = []
        self.context = None

    def _update_dialogues(self, text, log_prob=None):
        """
        Update the history dialogue with te given observation.
        dialogue is a tuple with index 0 from the others and the index 1 from self
        :param text: the current observed utterance text
        """

        if self.size > 0:
            while len(self.dialogues) >= self.size/2:
                self.dialogues.pop(0)
        dialogue = DialogueTurn(question_text=text, log_prob=log_prob)
        if self.dialogues and not self.dialogues[-1].complete:
            self.dialogues[-1].update(text, log_prob)
        else:
            self.dialogues.append(dialogue)

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
        self.temp_history = temp_history

    def get_history_vec(self):
        """
        Override from parent class to possibly add [SEP] token.
        """
        if not self.sep_last_utt or len(self.history_vecs) <= 1:
            return super().get_history_vec()

        history = deque(maxlen=self.max_len)
        for vec in self.history_vecs[:-1]:
            history.extend(vec)
            history.extend(self.delimiter_tok)
        history.extend([self.dict.end_idx])  # add [SEP] token
        history.extend(self.history_vecs[-1])

        return history

    def add_reply(self, text, log_prob=None):
        """
        Add your own response to the history.
        """
        self._update_raw_strings(text)
        if self.add_person_tokens:
            text = self._add_person_tokens(text, self.p2_token)
        # update history string
        self._update_strings(text)
        # update history vecs
        self._update_vecs(text)
        # update history dialogues
        self._update_dialogues(text)


class MultiDialogueHistory(DialogueHistory):

    def __init__(self, opt, history_class, field='text', **kwargs):
        self.sep_last_utt = opt.get('sep_last_utt', False)
        self.history_cls_func = history_class
        self.field = field
        self.lineages = []

    def add_lineage(self, text, history, log_prob=None):
        duplicated_history = copy.deepcopy(history)
        duplicated_history.add_reply(text, log_prob=log_prob)
        self.lineages.append(duplicated_history)

    def reset(self):
        """
        Clear the history.
        """
        self.lineages = []

    def update_history(self, obs: Message, temp_history: Optional[str] = None):
        for i, history in self.lineages:
            if temp_history:
                one_temp_history = temp_history[0]
            text = Message[self.field][i]
            history.update(text, one_temp_history)

    def get_history_str(self):
        if not self.lineages:
            return []
        return [hist.get_history_str for hist in self.lineages]

    def get_history_vec(self):
        """
        Return a vectorized version of the history.
        """
        if not self.lineages:
            return []
        return [hist.get_history_vec() for hist in self.lineages]

    def get_history_vec_list(self):
        """
        Return a list of history vecs.
        """
        if not self.lineages:
            return []
        return [hist.get_history_vec_list() for hist in self.lineages]

    def add_reply(self, texts):
        """
        Add your own response to the history.
        """
        for i, hist in enumerate(self.lineages):
            text = texts[i]
            self.lineages[i].add_reply(text)
