"""
Torch Classifier Agents extract answer text from the context by prediction the start_index and end_index.
"""

from abc import ABC, abstractmethod
from parlai.core.opt import Opt
from parlai.core.torch_agent import TorchAgent, Output, Optional, History
import parlai.utils.logging as logging
from parlai.utils.torch import PipelineHelper, total_parameters, trainable_parameters
from parlai.utils.fp16 import FP16SafeCrossEntropy
from parlai.core.metrics import AverageMetric
from parlai.core.message import Message
from collections import deque

import torch
import torch.nn as nn


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

class DialogueHistory(History):
    def __init__(self, opt, **kwargs):
        self.sep_last_utt = opt.get('sep_last_utt', False)
        super().__init__(opt, **kwargs)
        self.context = None

    def reset(self):
        """
        Clear the history.
        """
        self.history_raw_strings = []
        self.history_strings = []
        self.history_vecs = []
        self.context = None

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
        if not self.context:
            self.context = obs['context']
        text = obs['question']
        self._update_raw_strings(text)
        if self.add_person_tokens:
            text = self._add_person_tokens(
                obs[self.field], self.p1_token, self.add_p1_after_newln
            )
        # update history string
        self._update_strings(text)
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


class TorchExtractiveModel(nn.Module, ABC):
    """
    Abstract TorchGeneratorModel.

    This interface expects you to implement model with the following reqs:

    :attribute model.encoder:
        takes input returns tuple (enc_out, enc_hidden, attn_mask)

    :attribute model.classifier:
        takes decoder params and returns decoder outputs after attn

    :attribute model.output:
        takes decoder outputs and returns the answer_start logits and answer_end logit
    """

    def __init__(
            self
    ):
        super().__init__()
        self.num_labels = 2

    def forward(self, inputs, ys=None):
        """

        """
        assert ys is not None, "Greedy decoding in TGModel.forward no longer supported."
        outputs = self.encoder(
            **inputs,
        )
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits, sequence_output


class TorchSpanAgent(TorchAgent):
    """
    Abstract Classifier agent. Only meant to be extended.

    TorchClassifierAgent aims to handle much of the bookkeeping any classification
    model.
    """

    @staticmethod
    def add_cmdline_args(parser):
        """
        Add CLI args.
        """
        TorchAgent.add_cmdline_args(parser)
        parser = parser.add_argument_group('Torch Span Classifier Arguments')
        # interactive mode
        parser.add_argument(
            '--print-scores',
            type='bool',
            default=False,
            help='print probability of chosen class during ' 'interactive mode',
        )
        # miscellaneous arguments
        parser.add_argument(
            '--data-parallel',
            type='bool',
            default=False,
            help='uses nn.DataParallel for multi GPU',
        )
        # query maximum length
        parser.add_argument(
            '--history-maximum-length',
            type=int,
            default=63,
            help='maximum number of tokens allowed for quenry string',
        )
        # query maximum length
        parser.add_argument(
            '--query-maximum-length',
            type=int,
            default=63,
            help='maximum number of tokens allowed for quenry string',
        )
        # context maximum length
        parser.add_argument(
            '--context-maximum-length',
            type=int,
            default=385,
            help='maximum number of tokens allowed for quenry string',
        )

    @classmethod
    def history_class(cls):
        """
        Return the history class that this agent expects to use.

        Can be overriden if a more complex history is required.
        """
        return DialogueHistory

    def __init__(self, opt: Opt, shared=None):
        init_model, self.is_finetune = self._get_init_model(opt, shared)
        super().__init__(opt, shared)

        # set up model and optimizers
        self.query_truncate = opt['query_maximum_length']
        self.context_truncate = opt['context_maximum_length']
        self.history_truncate = opt['history_maximum_length']
        self.truncate = self.dict.tokenizer.max_len

        if shared:
            self.model = shared['model']
        else:
            self.model = self.build_model()
            self.criterion = self.build_criterion()
            if self.model is None or self.criterion is None:
                raise AttributeError(
                    'build_model() and build_criterion() need to return the model or criterion'
                )
            if init_model:
                logging.info(f'Loading existing model parameters from {init_model}')
                self.load(init_model)
            if self.use_cuda:
                if self.model_parallel:
                    ph = PipelineHelper()
                    ph.check_compatibility(self.opt)
                    self.model = ph.make_parallel(self.model)
                else:
                    self.model.cuda()
                if self.data_parallel:
                    self.model = torch.nn.DataParallel(self.model)
                self.criterion.cuda()

            train_params = trainable_parameters(self.model)
            total_params = total_parameters(self.model)
            logging.info(
                f"Total parameters: {total_params:,d} ({train_params:,d} trainable)"
            )

        if shared:
            # We don't use get here because hasattr is used on optimizer later.
            if 'optimizer' in shared:
                self.optimizer = shared['optimizer']
        elif self._should_initialize_optimizer():
            optim_params = [p for p in self.model.parameters() if p.requires_grad]
            self.init_optim(optim_params)
            self.build_lr_scheduler()

    def get_temp_history(self, observation) -> Optional[str]:
        """
        Return a string to temporarily insert into history.

        Intentionally overrideable so more complex models can insert temporary history
        strings, i.e. strings that are removed from the history after a single turn.
        """
        return None

    def compute_loss(self, batch, return_output=False):
        """
                Compute and return the loss for the given batch.

                Easily overridable for customized loss functions.

                If return_output is True, the full output from the call to self.model()
                is also returned, via a (loss, model_output) pair.
                """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')
        model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
        scores, preds, *_ = model_output
        score_view = scores.view(-1, scores.size(-1))
        loss = self.criterion(score_view, batch.label_vec.view(-1))
        loss = loss.view(scores.shape[:-1]).sum(dim=1)
        # save loss to metrics
        notnull = batch.label_vec.ne(self.NULL_IDX)
        target_tokens = notnull.long().sum(dim=-1)
        correct = ((batch.label_vec == preds) * notnull).sum(dim=-1)

        self.record_local_metric('loss', AverageMetric.many(loss, target_tokens))
        # actually do backwards loss
        loss = loss.sum()
        loss /= target_tokens.sum()  # average loss per token
        if return_output:
            return (loss, model_output)
        else:
            return loss

    def build_criterion(self):
        """
        Construct and return the loss function.

        By default torch.nn.CrossEntropyLoss.

        If overridden, this model should produce a sum that can be used for a per-token loss.
        """
        if self.fp16:
            return FP16SafeCrossEntropy(reduction='none')
        else:
            return torch.nn.CrossEntropyLoss(reduction='none')

    def reset_metrics(self):
        """
        Reset metrics for reporting loss and perplexity.
        """
        super().reset_metrics()

    def reset(self):
        """
        Clear internal states.
        """
        # assumption violation trackers
        self.__expecting_clear_history = False
        self.__expecting_to_reply = False

        self.observation = None
        self.history.reset()
        self.reset_metrics()

    def train_step(self, batch):
        """
        Train on a single batch of examples.
        """
        # helps with memory usage
        # note we want to use the opt's batchsize instead of the observed batch size
        # in case dynamic batching is in use
        self._init_cuda_buffer(self.opt['batchsize'], self.label_truncate or 256)
        self.model.train()
        self.zero_grad()

        try:
            loss = self.compute_loss(batch)
            self.backward(loss)
            self.update_params()
            oom_sync = False
        except RuntimeError as e:
            # catch out of memory exceptions during fwd/bck (skip batch)
            if 'out of memory' in str(e):
                oom_sync = True
                logging.error(
                    'Ran out of memory, skipping batch. '
                    'if this happens frequently, decrease batchsize or '
                    'truncate the inputs to the model.'
                )
                return Output()
            else:
                raise e

    def eval_step(self, batch):
        """
               Train on a single batch of examples.
               """
        if batch.text_vec is None:
            return

        self.model.eval()
        loss, model_output = self.compute_loss(batch, model_output=True)
        _, preds, *_ = model_output
        if batch.labels is None or self.opt['ignore_labels']:
            # interactive mode
            if self.opt.get('print_scores', False):
                preds = "Not yet implemented interactive mode"
        else:
            self.record_local_metric('loss', AverageMetric.many(loss))
            loss = loss.mean()
            return Output(preds)

    def _set_label_vec(self, obs, add_start, add_end, label_truncate):
        """
        Set the 'labels_vec' field in the observation.

        Useful to override to change vectorization behavior
        """
        # convert 'labels' or 'eval_labels' into vectors
        return obs

    # Tokenize our training dataset
    def _set_text_vec(self, obs, history, truncate):
        # Tokenize contexts and questions (as pairs of inputs)
        context_text = self.truncate_with_dic(obs['context'], self.context_truncate)
        history_text = self.truncate_with_dic(" ".join(history.history_strings[-2:]), self.history_truncate)
        question_text = self.truncate_with_dic(obs['question'], self.query_truncate)
        input_triplet = [question_text, context_text, history_text]
        encodings = self.dict.tokenizer.encode_plus(input_triplet, pad_to_max_length=True, max_length=512)
        context_encodings = self.dict.tokenizer.encode_plus(obs['context'])

        # Compute start and end tokens for labels using Transformers's fast tokenizers alignement methodes.
        # this will give us the position of answer span in the context text
        start_idx, end_idx = obs['answer_starts'], obs['answers_ends']
        start_positions_context = context_encodings.char_to_token(start_idx)
        end_positions_context = context_encodings.char_to_token(end_idx - 1)

        # here we will compute the start and end position of the answer in the whole example
        # as the example is encoded like this <s> question</s></s> context</s>
        # and we know the postion of the answer in the context
        # we can just find out the index of the sep token and then add that to position + 1 (+1 because there are two sep tokens)
        # this will give us the position of the answer span in whole example
        sep_idx = encodings['input_ids'].index(self.dict.tokenizer.sep_token_id)
        start_positions = start_positions_context + sep_idx + 1
        end_positions = end_positions_context + sep_idx + 1

        if end_positions > 512:
            start_positions, end_positions = 0, 0
        obs['text_vec'] = encodings['input_ids']
        obs['full_text_vec'] = encodings
        obs['answer_starts'] = start_positions
        obs['answers_ends'] = end_positions
        return obs

    def truncate_with_dic(self, text, truncate):
        ids = self.dict.txt2vec(text)
        if len(ids) <= truncate:
            return text
        else:
            ids = ids[:truncate]
            truncate_text = self.dict.tokenizer.decode(ids)
            return truncate_text

    def get_char_to_word_offset(self, text):
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
        return char_to_word_offset
