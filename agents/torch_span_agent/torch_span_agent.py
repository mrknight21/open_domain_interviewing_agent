"""
Torch Classifier Agents extract answer text from the context by prediction the start_index and end_index.
"""

from abc import ABC, abstractmethod
from parlai.core.opt import Opt
from parlai.core.torch_agent import TorchAgent, Output, Optional, History, Batch
import parlai.utils.logging as logging
from parlai.utils.torch import PipelineHelper, total_parameters, trainable_parameters, padded_tensor
from parlai.utils.fp16 import FP16SafeCrossEntropy
from parlai.core.metrics import AverageMetric
from parlai.core.message import Message
from collections import deque
import collections

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
        text = obs['question_text']
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

    def forward(self,
        inputs,
        output_attentions=None,
        output_hidden_states=None,
        ):
        """

        """
        outputs = self.encoder.transformer(
            **inputs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
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
            help='maximum number of tokens allowed for history string',
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
            help='maximum number of tokens allowed for context string',
        )
        # context maximum length
        parser.add_argument(
            '--doc_stride',
            type=int,
            default=128,
            help='When splitting up a long document into chunks, how much stride to take between chunks.',
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
        self.doc_stride = opt['doc_stride']

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
        start_logits, end_logits, sequence_output = self.model(self._model_input(batch))
        start_positions = batch.get('start_positions', None)
        end_positions = batch.get('end_positions', None)
        output_start_positions, output_end_positions = (torch.argmax(start_logits, dim=1).unsqueeze(0), torch.argmax(end_logits, dim=1).unsqueeze(0))
        end_start_pairs = torch.cat((output_start_positions, output_end_positions), 1).cpu().data.numpy()
        output_text = []
        output_ids = []
        for i in range(end_start_pairs.shape[0]):
            pair = end_start_pairs[i]
            if pair[0] <= pair[1]:
                output_id = batch.encoding['input_ids'][i][pair[0]: pair[1]+1]
                text = self.dict.tokenizer.decode(output_id)
                output_text.append(text)
                output_ids.append(output_id)
            else:
                output_text.append("")
        model_output = {"output_start_positions": output_start_positions, "output_end_positions": output_end_positions, "span_text": output_text}
        total_loss = None
        if batch.get('start_positions', None) is not None:
            # If we are on multi-GPU, split add a dimension
            #
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            #
            start_loss = self.criterion(start_logits, start_positions)
            end_loss = self.criterion(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        self.record_local_metric('total_loss', AverageMetric.many(total_loss, [len(batch.observations)]))
        self.record_local_metric('start_loss', AverageMetric.many(start_loss, [len(batch.observations)]))
        self.record_local_metric('end_loss', AverageMetric.many(end_loss, [len(batch.observations)]))
        if return_output:
            return (total_loss, model_output)
        else:
            return total_loss

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
        # self._init_cuda_buffer(self.opt['batchsize'], self.label_truncate or 256)
        self.model.train()
        self.zero_grad()

        try:
            loss, model_output = self.compute_loss(batch, return_output=True)
            self.backward(loss)
            self.update_params()
            oom_sync = False
            # return Output(**model_output)
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
        obs['label_vec'] = [obs['text_vec'][i][start: end+1]
                            for i, (start, end) in enumerate(zip(obs['answer_starts'], obs['answer_ends']))]
        return obs

    # Tokenize our training dataset
    def _set_text_vec(self, obs, history, truncate, is_training=True):
        # Tokenize contexts and questions (as pairs of inputs)
        text_vecs = []
        full_text_vecs = []
        start_positions = []
        end_positions = []
        ans_text = obs['labels']
        history_text = self.truncate_with_dic(" ".join(history.history_strings[:-1]), self.history_truncate, latest=True)
        question_text = self.truncate_with_dic(obs['question_text'], self.query_truncate)
        query_tokens = self.dict.tokenizer.tokenize(question_text)

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(obs['doc_tokens']):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = self.dict.tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and obs['is_impossible']:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not obs['is_impossible']:
            tok_start_position = orig_to_tok_index[obs['start_position']]
            if obs['end_position'] < len(obs['doc_tokens']) - 1:
                tok_end_position = orig_to_tok_index[obs['end_position'] + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = self.improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, self.dict.tokenizer,
                ans_text)

        # The -3 accounts for [CLS], [SEP] and [SEP] and [SEP]
        max_tokens_for_doc = self.truncate - len(query_tokens) - self.history_truncate - 4

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, self.doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            # context_tokens = all_doc_tokens[doc_span[0]:doc_span[1]]
            context_text = self.slice_text_with_token_index(obs['context'], doc_span[0], doc_span[1])
            if history_text:
                context_text = context_text + " " + history_text
            input_triplet = [question_text, context_text]
            encodings = self.dict.tokenizer.encode_plus(*input_triplet,
                                                        pad_to_max_length=True,
                                                        add_special_tokens=True,
                                                        padding=True,
                                                        max_length=self.truncate,
                                                        return_attention_mask=True,
                                                        truncation=True,
                                                        return_tensors='pt')
            start_position = None
            end_position = None
            if is_training and not obs['is_impossible']:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and obs['is_impossible']:
                start_position = 0
                end_position = 0

            text_vecs.extend(encodings['input_ids'])
            full_text_vecs.append(encodings)
            start_positions.append(start_position)
            end_positions.append(end_position)

        obs['text_vec'] = text_vecs
        obs['full_text_vec'] = full_text_vecs
        obs['answer_starts'] = start_positions
        obs['answer_ends'] = end_positions
        return obs

    def batchify(self, obs_batch, sort=False):

        if len(obs_batch) == 0:
            return Batch(batchsize=0)

        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if self.is_valid(ex)]

        if len(valid_obs) == 0:
            return Batch(batchsize=0)

        valid_inds, exs = zip(*valid_obs)
        encodings = [text_vec for text_vec in obs_batch[0]['full_text_vec']]
        input_ids = []
        token_type_ids = []
        attention_mask = []
        for i, encoding in enumerate(encodings):
            input_ids.append(encoding['input_ids'])
            token_type_ids.append(encoding['token_type_ids'])
            attention_mask.append(encoding['attention_mask'])
        input_ids = torch.cat(input_ids)
        token_type_ids = torch.cat(token_type_ids)
        attention_mask = torch.cat(attention_mask)
        xs, x_lens = self._pad_tensor([text_vec for batch in obs_batch for text_vec in batch['text_vec']])
        start_positions = torch.LongTensor([batch['answer_starts'] for batch in obs_batch])
        end_positions = torch.LongTensor([batch['answer_ends'] for batch in obs_batch])
        labels = [batch['labels'] for batch in obs_batch]
        label_vec, label_vec_len = self._pad_tensor([vec for batch in obs_batch for vec in batch['label_vec']])
        ys, y_lens = start_positions, len(start_positions)
        if self.use_cuda:
            start_positions = start_positions.cuda()
            end_positions = end_positions.cuda()
            input_ids = input_ids.cuda()
            token_type_ids = token_type_ids.cuda()
            attention_mask = attention_mask.cuda()

        batch_encoding = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}

        return Batch(
            batchsize=len(valid_inds),
            encoding=batch_encoding,
            text_vec=xs,
            text_lengths=x_lens,
            label_vec=label_vec,
            label_lengths=y_lens,
            labels_text=labels,
            valid_indices=valid_inds,
            observations=exs,
            start_positions=start_positions,
            end_positions=end_positions
        )

    def _dummy_batch(self, batchsize, maxlen):
        """
        Create a dummy batch.

        This is used to preinitialize the cuda buffer, or otherwise force a
        null backward pass after an OOM.

        If your model uses additional inputs beyond text_vec and label_vec,
        you will need to override it to add additional fields.
        """
        text_vec = (
            torch.arange(1, maxlen + 1)  # need it as long as specified
            .clamp(max=3)  # cap at 3 for testing with tiny dictionaries
            .unsqueeze(0)
            .expand(batchsize, maxlen)
            .cuda()
        )
        # label vec has two tokens to make it interesting, but we we can't use the
        # start token, it's reserved.
        label_vec = (
            torch.LongTensor([self.END_IDX, self.NULL_IDX])
            .unsqueeze(0)
            .expand(batchsize, 2)
            .cuda()
        )
        return Batch(
            text_vec=text_vec, label_vec=label_vec, text_lengths=[maxlen] * batchsize
        )

    def _init_cuda_buffer(self, batchsize, maxlen, force=False):
        """
        Pre-initialize CUDA buffer by doing fake forward pass.

        This is also used in distributed mode to force a worker to sync with others.
        """
        if self.use_cuda and (force or not hasattr(self, 'buffer_initialized')):
            try:
                self._control_local_metrics(disabled=True)
                loss = 0 * self.compute_loss(self._dummy_batch(batchsize, maxlen))
                self._control_local_metrics(enabled=True)
                self._temporarily_disable_local_metrics = False
                self.backward(loss)
                self.buffer_initialized = True
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    m = (
                        'CUDA OOM: Lower batch size (-bs) from {} or lower '
                        ' max sequence length (-tr) from {}'
                        ''.format(batchsize, maxlen)
                    )
                    raise RuntimeError(m)
                else:
                    raise e

    def _pad_tensor(self, items):
        """
        Override to always set fp16friendly to False.
        """
        return padded_tensor(
            items, pad_idx=self.NULL_IDX, use_cuda=self.use_cuda, fp16friendly=False
        )

    def piece_word_char_to_token(self, tokens, start_index, answer):
        token_start_index = []
        for index, ans in zip(start_index, answer):
            char_index = 0
            for token in tokens:
                subword = False
                if '##' in token:
                    subword = True
                    word = token.replace('##', '')

    def truncate_with_dic(self, text, truncate, latest=False):
        if not text:
            return ""
        tokens = text.split(" ")
        if latest:
            tokens = tokens.reverse()
        tokens_in_range = []
        tokens_count = 0
        for (i, token) in enumerate(tokens):
            if tokens_count > truncate:
                break
            tokens_in_range.append(token)
            sub_tokens = self.dict.tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tokens_count += 1
        if latest:
            tokens_in_range = tokens_in_range.reverse()
        return " ".join(tokens_in_range)

    def slice_text_with_token_index(self, text, start, end):
        if not text:
            return ""
        tokens = text.split(" ")
        tokens_in_range = []
        subwords_count = 0
        for (i, token) in enumerate(tokens):
            if subwords_count > end:
                break
            elif subwords_count >= start:
                tokens_in_range.append(token)
            sub_tokens = self.dict.tokenizer.tokenize(token)
            subwords_count += len(sub_tokens)
        if tokens_in_range:
            return " ".join(tokens_in_range)
        else:
            return ""

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

    def improve_answer_span(self, doc_tokens, input_start, input_end, tokenizer,
                             orig_answer_text):
        tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)

    def _check_is_max_context(self, doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""

        # Because of the sliding window approach taken to scoring documents, a single
        # token can appear in multiple documents. E.g.
        #  Doc: the man went to the store and bought a gallon of milk
        #  Span A: the man went to the
        #  Span B: to the store and bought
        #  Span C: and bought a gallon of
        #  ...
        #
        # Now the word 'bought' will have two scores from spans B and C. We only
        # want to consider the score with "maximum context", which we define as
        # the *minimum* of its left and right context (the *sum* of left and
        # right context will always be the same, of course).
        #
        # In the example the maximum context for 'bought' would be span C since
        # it has 1 left context and 3 right context, while span B has 4 left context
        # and 0 right context.
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index
