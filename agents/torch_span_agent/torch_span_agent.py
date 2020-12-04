"""
Torch Classifier Agents extract answer text from the context by prediction the start_index and end_index.
"""

from abc import ABC, abstractmethod
from parlai.core.opt import Opt
from parlai.core.torch_agent import TorchAgent, Output, Optional, History, Batch
import parlai.utils.logging as logging
from parlai.utils.distributed import is_distributed, sync_parameters
from parlai.utils.torch import PipelineHelper, total_parameters, trainable_parameters, padded_tensor
from parlai.utils.fp16 import FP16SafeCrossEntropy
from parlai.core.metrics import AverageMetric
from parlai.core.message import Message
from parlai_internal.utilities import util

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
        if "text" in obs and obs["text"] is not None:
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
        init_model, is_finetune = self._get_init_model(opt, shared)
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
                states = self.load(init_model)
            else:
                states = {}
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

            if self.fp16:
                self.model = self.model.half()

        if shared is not None:
            if 'optimizer' in shared:
                self.optimizer = shared['optimizer']
        elif self._should_initialize_optimizer():
            # do this regardless of share state, but don't
            self.init_optim(
                [p for p in self.model.parameters() if p.requires_grad],
                optim_states=states.get('optimizer'),
                saved_optim_type=states.get('optimizer_type'),
            )
            self.build_lr_scheduler(states, hard_reset=is_finetune)

        if shared is None and is_distributed():
            device_ids = None if self.model_parallel else [self.opt['gpu']]
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=device_ids, broadcast_buffers=False
            )

        self.reset()


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
        output_start_positions, output_end_positions = torch.argmax(start_logits, dim=1), torch.argmax(end_logits, dim=1)
        end_start_pairs = torch.stack((output_start_positions, output_end_positions), dim=1).cpu().data.numpy()
        output_text = []
        combined_pair_conf = []
        batch_output_text = []
        for i in range(end_start_pairs.shape[0]):
            pair = end_start_pairs[i]
            start_confidence = start_logits[i][pair[0]]
            end_confidence = end_logits[i][pair[1]]
            pair_confidence = start_confidence + end_confidence
            combined_pair_conf.append(pair_confidence)
            if pair[0] <= pair[1]:
                output_id = batch.encoding['input_ids'][i][pair[0]: pair[1]+1]
                text = self.dict.tokenizer.decode(output_id).replace("[CLS]", '')
                if text == "":
                    output_text.append("[CLS]")
                else:
                    output_text.append(text)
            else:
                output_text.append("[CLS]")

        total_loss = None
        losses = []
        start_losses = []
        end_losses = []
        correct_span_nums = []
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
            for doc_indexes in batch['batch_indexes_map']:
                cur_loss = 0
                index_start, index_end = doc_indexes[0], doc_indexes[-1]+1
                cur_output_tex = output_text[index_start: index_end]
                cur_pair_conf = combined_pair_conf[index_start: index_end]
                max_conf_index = cur_pair_conf.index(max(cur_pair_conf))
                batch_output_text.append(cur_output_tex[max_conf_index])

                cur_start_logits = start_logits[index_start: index_end]
                cur_end_logits = end_logits[index_start: index_end]
                start_loss = self.criterion(cur_start_logits, torch.flatten(start_positions[index_start: index_end])).mean()
                end_loss = self.criterion(cur_end_logits, torch.flatten(end_positions[index_start: index_end])).mean()
                cur_loss = (start_loss + end_loss) / 2
                start_corrects = output_start_positions[index_start: index_end] == start_positions[index_start: index_end]
                end_corrects = output_end_positions[index_start: index_end] == end_positions[index_start: index_end]
                correct_span_num = start_corrects * end_corrects
                correct_span_num = correct_span_num.sum()
                losses.append(cur_loss)
                start_losses.append(start_loss)
                end_losses.append(end_loss)
                correct_span_nums.append(correct_span_num)
            batches_count = [1]*len(batch['batch_indexes_map'])
            doc_count = [len(doc_indexes) for doc_indexes in batch['batch_indexes_map']]
            self.record_local_metric('total_loss', AverageMetric.many(losses, batches_count))
            self.record_local_metric('start_loss', AverageMetric.many(start_losses, batches_count))
            self.record_local_metric('end_loss', AverageMetric.many(end_losses, batches_count))
            self.record_local_metric('span_acc', AverageMetric.many(correct_span_nums, doc_count))

        model_output = {"output_start_positions": output_start_positions, "output_end_positions": output_end_positions,
                        "text": batch_output_text}
        total_loss = torch.stack(losses).mean()
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
        if batch.batch_indexes_map is None:
            return
        else:
            bsz = len(batch.batch_indexes_map)
        self.model.eval()
        loss, model_output = self.compute_loss(batch, return_output=True)
        preds = model_output['text']
        return Output(preds)


    def _set_label_vec(self, obs, add_start, add_end, label_truncate):
        """
        Set the 'labels_vec' field in the observation.

        Useful to override to change vectorization behavior
        """
        # convert 'labels' or 'eval_labels' into vectors
        if 'answer_starts' not in obs or 'answer_ends' not in obs:
            return None
        obs['label_vec'] = [obs['text_vec'][i][start: end+1]
                            for i, (start, end) in enumerate(zip(obs['answer_starts'], obs['answer_ends']))]
        return obs

    # Tokenize our training dataset
    def _set_text_vec(self, obs, history, truncate, is_training=True):
        # Tokenize contexts and questions (as pairs of inputs)
        if 'text' not in obs:
            return obs
        start_positions = []
        end_positions = []
        ans_text = obs.get('single_label_text', None)
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
            (tok_start_position, tok_end_position) = util.improve_answer_span(
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

        question_texts = []
        context_texts = []
        text_vecs = []

        if len(doc_spans) > 1:
            logging.info('Chuncking document with {} tokens shift'.format(self.doc_stride))
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            # context_tokens = all_doc_tokens[doc_span[0]:doc_span[1]]
            context_text = self.slice_text_with_token_index(obs['context'], doc_span[0], doc_span[1])
            if history_text:
                context_text = context_text + " " + history_text

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
            text_vec = self.dict.tokenizer.encode_plus(question_text, context_text,
                                                        pad_to_max_length=True,
                                                        add_special_tokens=True,
                                                        padding=True,
                                                        max_length=self.truncate,
                                                        return_attention_mask=True,
                                                        truncation=True,
                                                        return_tensors='pt')['input_ids'][0]
            text_vecs.append(text_vec)
            question_texts.append(question_text)
            context_texts.append(context_text)
            start_positions.append(start_position)
            end_positions.append(end_position)

        full_text_dict ={'question_texts': question_texts, 'context_texts': context_texts}

        obs['text_vec'] = text_vecs
        obs['full_text_dict'] = full_text_dict
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
        batch_indexes_map = []
        start_positions = []
        end_positions = []
        question_texts = []
        context_texts = []
        labels = []
        label_vec = []
        cur_index = 0
        for i in valid_inds:
            batch = obs_batch[i]
            doc_indexes = []
            start_positions.extend(batch['answer_starts'])
            end_positions.extend(batch['answer_ends'])
            labels.append(batch.get('labels', batch.get('eva_labels')))
            label_vec.extend(batch['label_vec'])
            question_texts.extend(batch['full_text_dict']['question_texts'])
            context_texts.extend(batch['full_text_dict']['context_texts'])
            doc_num = len(batch['answer_starts'])
            for _ in range(doc_num):
                doc_indexes.append(cur_index)
                cur_index += 1
            batch_indexes_map.append(doc_indexes)

        encodings = self.dict.tokenizer(question_texts, context_texts,
                                        pad_to_max_length=True,
                                        add_special_tokens=True,
                                        padding=True,
                                        max_length=self.truncate,
                                        return_attention_mask=True,
                                        truncation=True,
                                        return_tensors='pt')
        start_positions = torch.LongTensor(start_positions)
        end_positions = torch.LongTensor(end_positions)
        # xs, x_lens = self._pad_tensor(xs)
        # label_vec, label_vec_len = self._pad_tensor(label_vec)
        # ys, y_lens = start_positions, len(start_positions)

        if self.use_cuda:
            start_positions = start_positions.cuda()
            end_positions = end_positions.cuda()
            encodings['input_ids'] = encodings['input_ids'].cuda()
            encodings['token_type_ids'] = encodings['token_type_ids'].cuda()
            encodings['attention_mask'] = encodings['attention_mask'].cuda()

        return Batch(
            batchsize=len(valid_inds),
            encoding=encodings,
            label_vec=label_vec,
            labels_text=labels,
            valid_indices=valid_inds,
            observations=exs,
            start_positions=start_positions,
            end_positions=end_positions,
            batch_indexes_map=batch_indexes_map
        )

    def _pad_tensor(self, items):
        """
        Override to always set fp16friendly to False.
        """
        return padded_tensor(
            items, pad_idx=self.dict.pad_idx, use_cuda=self.use_cuda, fp16friendly=False
        )

    def _model_input(self, batch):
        """
        Override to pass in text lengths.
        """
        return (batch.encoding)

    def _encoder_input(self, batch):
        return (batch.text_vec)

    def truncate_with_dic(self, text, truncate, latest=False):
        if not text:
            return ""
        tokens = text.split(" ")
        if latest:
            tokens.reverse()
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
            tokens_in_range.reverse()
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




