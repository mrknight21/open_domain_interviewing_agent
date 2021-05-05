from parlai_internal.agents.interviewer.interviewer import InterviewerAgent
from parlai_internal.utilities.flow_lstm_util import constants
from parlai.agents.hugging_face.gpt2 import Gpt2DictionaryAgent, GPT2Decoder, HFGPT2Model
from parlai.core.torch_agent import Optional, Batch, Output
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.utils.misc import warn_once

from typing import List
import copy

SPECIAL_TOKENS = {"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>",
                  "QUESST": constants.QUESST, "QUESEN": constants.QUESEN,
                  "TITLEST": constants.TITLEST, "TITLEEN": constants.TITLEEN,
                  "SECST": constants.SECST, "SECEN": constants.SECEN,
                  "BGST": constants.BGST, "BGEN": constants.BGEN,
                  "ANSST": constants.ANSST, "ANSEN": constants.ANSEN}

NO_OP = "x"

class GptInterviewerDictionaryAgent(Gpt2DictionaryAgent):

    def add_additional_special_tokens(self, additional_special_tokens: List[str]):
        """
        Add additional special tokens to the dictionary.
        """
        self.additional_special_tokens = additional_special_tokens
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': additional_special_tokens}
        )
        for tok in self.additional_special_tokens:
            self.add_token(tok)

    def _define_special_tokens(self, opt):
        if opt["add_special_tokens"]:
            # Add addtional start/end/pad tokens
            self.tokenizer.add_special_tokens(SPECIAL_TOKENS)
            self.start_token = SPECIAL_TOKENS["bos_token"]
            self.end_token = SPECIAL_TOKENS["eos_token"]
            self.null_token = SPECIAL_TOKENS["pad_token"]
        else:
            # Only special token is end of text
            self.start_token = NO_OP  # hack, we cut off the start token
            self.end_token = "<|endoftext|>"
            self.null_token = "<|endoftext|>"

    def override_special_tokens(self, opt):
        # define special tokens
        self._define_special_tokens(opt)
        # now override
        self.start_idx = self.tokenizer.convert_tokens_to_ids([self.start_token])[0]
        self.end_idx = self.tokenizer.convert_tokens_to_ids([self.end_token])[0]
        self.null_idx = self.tokenizer.convert_tokens_to_ids([self.null_token])[0]
        # set tok2ind for special tokens
        self.tok2ind[self.end_token] = self.end_idx
        self.tok2ind[self.start_token] = self.start_idx
        self.tok2ind[self.null_token] = self.null_idx
        # set ind2tok for special tokens
        self.ind2tok[self.end_idx] = self.end_token
        self.ind2tok[self.start_idx] = self.start_token
        self.ind2tok[self.null_idx] = self.null_token

class GptInterviewerAgent(InterviewerAgent):

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        agent = parser.add_argument_group("Gpt2 Args")
        agent.add_argument(
            "--gpt2-size",
            type=str,
            default="small",
            choices=["small", "medium", "large", "xl", "distilgpt2"],
            help="Which size model to initialize.",
        )
        agent.add_argument(
            "--add-special-tokens",
            type="bool",
            default=True,
            help="Add special tokens (like PAD, etc.). If False, "
            "Can only use with batch size 1.",
        )
        agent.add_argument(
            "--add-start-token",
            type="bool",
            default=False,
            help="Add start tokens when finetuning.",
        )
        parser.set_defaults(
            text_truncate=768,
            label_truncate=256,
            dict_maxexs=0,  # skip building dictionary
        )
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        warn_once("WARNING: this model is in beta and the API is subject to change.")
        return agent

    def __init__(self, opt: Opt, shared=None):
        if not opt["add_special_tokens"] and opt.get('batchsize', 1) > 1:
            # *** STOP ***
            # You may be a future researcher who has stumbled upon this odd
            # restriction, and is tempted to comment this out. After all, the
            # code still runs when it's uncommented, why shouldn't you?
            # You should know this has serious implications, as gpt2 doesn't have
            # padding tokens. This is incompatible with ParlAI's batching,
            # which puts conversations of different length in the same
            # batch. Without a padding token, nonsense will be inserted into
            # the context, and the generations & PPL you get will be wrong.
            raise RuntimeError(
                "If using batchsize > 1, --add-special-tokens must be True."
            )
        if not opt["add_special_tokens"] and opt["add_start_token"]:
            raise RuntimeError(
                "--add-start-token true requires --add-special-tokens true"
            )
        super().__init__(opt, shared)
        if hasattr(self.model, "module"):
            self.START_IDX = self.model.module.START_IDX
            self.END_IDX = self.model.module.END_IDX
            self.NULL_IDX = self.model.module.NULL_IDX
        else:
            self.START_IDX = self.model.START_IDX
            self.END_IDX = self.model.END_IDX
            self.NULL_IDX = self.model.NULL_IDX

    @staticmethod
    def dictionary_class():
        """
        Return the dictionary class that this agent expects to use.

        Can be overriden if a more complex dictionary is required.
        """
        return GptInterviewerDictionaryAgent

    def build_model(self, states=None):
        """
        Build and return model.
        """
        return HFGPT2Model(self.opt, self.dict)


    def _set_text_vec(self, obs, history, truncate, is_training=True):
        history_dialogues = [self.history.dialogues]
        # original_answer = obs['text']
        # original_question = obs['single_label_text']
        retvals = []
        if len(self.diverged_dialogues.lineages) > 0:
            history_dialogues += [lineage.dialogues for lineage in self.diverged_dialogues.lineages if not lineage.freeze]
        for dialogues in history_dialogues:
            retval = copy.copy(obs)
            if dialogues:
                retval.force_set('text', dialogues[-1].answer)
            tokenized_data = self.tokenize_from_history(retval, dialogues)
            retval['text_vec'] = tokenized_data
            retvals.append(retval)
        #The master history lineage
        obs['text_vec'] = retvals[0]['text_vec']
        if len(retvals) > 1:
            obs['diverged_obs'] = retvals[1:]
        labels_with_special_tokens = []
        for l in obs.get('labels', obs.get('eval_labels', [])):
            if l == "":
                labels_with_special_tokens.append("")
            else:
                labels_with_special_tokens.append(l)
        if 'labels' in obs:
            obs.force_set('labels', labels_with_special_tokens)
        elif 'eval_labels' in obs:
            obs.force_set('eval_labels', labels_with_special_tokens)
        return obs


    def get_preprocessed_batches(self, obs_batch, valid_inds):
        batch = super(InterviewerAgent, self).batchify(obs_batch)
        return batch

    def tokenize_from_history(self, item, dialogues=None):
        history = self.history
        if not dialogues:
            dialogues = history.dialogues
        background_tokens = history.background_tokens
        if not background_tokens:
            if history.title:
                background_tokens += [constants.TITLEST] + self.dict.tokenizer.tokenize(history.title) + [constants.TITLEEN]
            if history.background:
                background_tokens += [constants.BGST] + self.dict.tokenizer.tokenize(history.background)[:constants.MAX_BACKGROUND] + [constants.BGEN]
            if history.section_title:
                background_tokens += [constants.SECST] + self.dict.tokenizer.tokenize(history.section_title) + [constants.SECEN]
            self.history.background_tokens = background_tokens
        qas = []
        if len(dialogues) > 0:
            for turn in dialogues:
                background_tokens += [constants.QUESST] + turn.question + [constants.QUESEN]
                background_tokens += [constants.ANSST] + turn.answer + [constants.ANSEN]
                qas.append((turn.question, turn.answer))
        strings_to_tokenize = " ".join(background_tokens)
        text_vec = self.dict.tokenizer.text2vec(strings_to_tokenize)
        return text_vec

    def _encoder_input(self, batch):
        return (batch.text_vec,)