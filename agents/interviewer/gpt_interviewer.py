from parlai_internal.agents.interviewer.interviewer import InterviewerAgent
from parlai_internal.utilities.flow_lstm_util import constants
from parlai.agents.hugging_face.gpt2 import Gpt2DictionaryAgent,  HFGPT2Model
from parlai.core.torch_agent import Output
import parlai.utils.logging as logging
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.utils.misc import warn_once

from typing import TypeVar, List, Dict, Optional, Tuple, Set, Iterable
import copy
import torch

SPECIAL_TOKENS = {"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>",
                  "additional_special_tokens": [constants.QUESST, constants.QUESEN,
                                                constants.TITLEST, constants.TITLEEN,
                                                constants.SECST, constants.SECEN,
                                                constants.BGST, constants.BGEN,
                                                constants.ANSST, constants.ANSEN]}

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
            text_truncate=738,
            label_truncate=30,
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

    def _set_label_vec(self, obs, add_start, add_end, truncate):
        """
        Set the 'labels_vec' field in the observation.

        Useful to override to change vectorization behavior
        """
        obs = super(InterviewerAgent, self)._set_label_vec(obs, add_start, add_end, truncate)
        return obs

    def get_preprocessed_batches(self, obs_batch, valid_inds):
        batch = super(InterviewerAgent, self).batchify(obs_batch)
        return batch

    def tokenize_from_history(self, item, dialogues=None):
        history = self.history
        if not dialogues:
            dialogues = history.dialogues
        background_string = history.background_string
        if not background_string:
            background_string = ""
            if history.title:
                background_string += constants.TITLEST + history.title + constants.TITLEEN
            if history.section_title:
                background_string += constants.SECST + history.section_title + constants.SECEN
            if history.background:
                background_string += constants.BGST + " ".join(history.background.split(" ")[:constants.MAX_BACKGROUND]) + constants.BGEN
            self.history.background_string = background_string
        qas = []
        if len(dialogues) > 0:
            for turn in dialogues:
                background_string += constants.QUESST + turn.question + constants.QUESEN
                background_string += constants.ANSST + turn.answer + constants.ANSEN
                qas.append((turn.question, turn.answer))
        text_vec = self.dict.txt2vec(background_string)
        return text_vec

    def _encoder_input(self, batch):
        return (batch.text_vec,)

    def _model_input(self, batch):
        return batch.text_vec

    def build_criterion(self):
        return super(InterviewerAgent, self).build_criterion()

    def rl_eval_step(self, batch):
        div_batch = batch.get('diverged_batch', None)
        if not div_batch:
            div_batch = batch
        token_losses = None
        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            loss, model_output = self.compute_loss(batch, return_output=True)
            if self.output_token_losses:
                token_losses = self._construct_token_losses(
                    batch.label_vec, model_output
                )
        preds = None
        maxlen = self.question_truncate or 30
        if self.eva_sample:
            preds, text, scores = self.sample(div_batch, latest_turn_only=True)
        else:
            preds, text, scores = self.predict(div_batch, latest_turn_only=True)
        retval = Output(text[:1], log_probs=scores[:1], episode_end=[batch.episode_end], ques_len=[len(preds[0])-1],  diverged_outputs=[[(t, scores[i], len(preds[i])-1) for i, t in enumerate(text[1:])]])
        return retval

    def rl_train_step(self, batch):
        maxlen = self.question_truncate or 30
        preds, text, nll = self.sample(batch, latest_turn_only=True)
        if self.rl_baseline_method == "self_critic":
            g_preds, g_text, g_scores = self.predict(batch, latest_turn_only=True, no_grad=True)
            retval = Output(text[:1], log_probs=nll[:1], episode_end=[batch['episode_end']],
                            ques_len=[len(preds[0]) - 1],
                            diverged_outputs=[[(t, nll[i], len(preds[i]) - 1) for i, t in enumerate(text[1:])]],
                            greedy_master_output=g_text[:1],
                            greedy_output=[[t for t in g_text[1:]]])
        else:
            retval = Output(text[:1], log_probs=nll[:1], episode_end=[batch['episode_end']], ques_len=[len(preds[0])-1],  diverged_outputs=[[(t, nll[i], len(preds[i])-1) for i, t in enumerate(text[1:])]])
        return retval

    def sample(self, batch, return_pair_level=False, latest_turn_only=True, train=True):
        self.opt['inference'] = 'nucleus'
        if batch.text_vec is None and batch.image is None:
            return
        if batch.text_vec is not None:
            bsz = batch.text_vec.size(0)
        else:
            bsz = len(batch.image)
        maxlen = self.label_truncate or 256
        if train:
            self.opt['topp'] = 1.0
            self.beam_min_length = 0
        beam_preds_scores, beams = self._generate(batch, self.beam_size, maxlen)
        preds, scores = zip(*beam_preds_scores)
        self._add_generation_metrics(batch, preds)

        # bsz x beamsize
        beam_texts: List[List[Tuple[str, float]]] = []
        for beam in beams:
            beam_texts.append([])
            for tokens, score in beam.get_rescored_finished():
                try:
                    beam_texts[-1].append((self._v2t(tokens), score.item()))
                except KeyError:
                    logging.error("Decoding error: %s", tokens)
                    continue
        text = [self._v2t(p) for p in preds] if preds is not None else None
        preds = [[int(id.detach().cpu())  for id in cov if id not in [self.START_IDX, self.END_IDX]] for cov in preds]
        return preds, text, scores

    def predict(self, batch, beam_size=1, return_pair_level=False, latest_turn_only=False, no_grad=True):
        self.opt['inference'] = 'greedy'
        if batch.text_vec is None and batch.image is None:
            return
        if batch.text_vec is not None:
            bsz = batch.text_vec.size(0)
        else:
            bsz = len(batch.image)
        maxlen = self.label_truncate or 256
        if no_grad:
            with torch.no_grad():
                beam_preds_scores, beams = self._generate(batch, self.beam_size, maxlen)
        else:
            beam_preds_scores, beams = self._generate(batch, self.beam_size, maxlen)
        preds, scores = zip(*beam_preds_scores)
        self._add_generation_metrics(batch, preds)
        # bsz x beamsize
        beam_texts: List[List[Tuple[str, float]]] = []
        for beam in beams:
            beam_texts.append([])
            for tokens, score in beam.get_rescored_finished():
                try:
                    beam_texts[-1].append((self._v2t(tokens), score.item()))
                except KeyError:
                    logging.error("Decoding error: %s", tokens)
                    continue
        text = [self._v2t(p) for p in preds] if preds is not None else None
        preds = [[int(id.detach().cpu()) for id in cov if id not in [self.START_IDX, self.END_IDX]] for cov in preds]
        return preds, text, scores