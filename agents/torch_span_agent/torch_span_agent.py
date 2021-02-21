"""
Torch Classifier Agents extract answer text from the context by prediction the start_index and end_index.
"""

from abc import ABC
from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.torch_agent import TorchAgent, Output, Optional, History, Batch
import parlai.utils.logging as logging
from parlai.utils.distributed import is_distributed
from parlai.utils.torch import PipelineHelper, total_parameters, trainable_parameters, padded_tensor
from parlai.utils.fp16 import FP16SafeCrossEntropy
from parlai_internal.utilities import util
from parlai.core.metrics import AverageMetric, F1Metric
from parlai.core.message import Message


from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
)
from transformers.data.processors.squad import (
    SquadFeatures,
    SquadResult,
    _new_check_is_max_context,
    _improve_answer_span,
    MULTI_SEP_TOKENS_TOKENIZERS_SET,
    whitespace_tokenize
)
from transformers.data.metrics.squad_metrics import (
    _get_best_indexes,
    get_final_text,
    _compute_softmax
)
from transformers.tokenization_utils_base import TruncationStrategy


from collections import deque, namedtuple, OrderedDict
import numpy as np
import torch
import torch.nn as nn


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
        self.history_dialogues = []
        self.history_strings = []
        self.history_vecs = []
        self.context = None

    def _update_dialogues(self, text):
        """
        Update the history dialogue with te given observation.
        dialogue is a tuple with index 0 from the others and the index 1 from self
        :param text: the current observed utterance text
        """
        if self.size > 0:
            while len(self.history_dialogues) >= self.size/2:
                self.history_dialogues.pop(0)
        dialogue = [text, None]
        if self.history_dialogues and self.history_dialogues[-1][1] is None:
            self.history_dialogues[-1][1] = text
        else:
            self.history_dialogues.append(dialogue)

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

    def add_reply(self, text):
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

    def __init__(self, opt):
        super().__init__()
        self.config_key = self.get_config_key(opt)
        self.hfconfig = AutoConfig.from_pretrained(self.config_key)
        self.use_cuda = not opt["no_cuda"] and torch.cuda.is_available()
        self.init_model(opt)

    def init_model(self, opt):
        self.transformer = AutoModelForQuestionAnswering.from_pretrained(
            self.config_key, config=self.hfconfig
        )

    def forward(self,
        inputs,
        output_attentions=None,
        output_hidden_states=None,
        ):

        outputs = self.transformer(
            **inputs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        return outputs

    def get_config_key(self, opt):
        raise NotImplementedError('not implemented for this class')


class TorchSpanAgent(TorchAgent):
    """
    Abstract Classifier agent. Only meant to be extended.

    TorchClassifierAgent aims to handle much of the bookkeeping any classification
    model.
    """

    @classmethod
    def add_cmdline_args(cls, parser, partial_opt: Optional[Opt] = None) -> ParlaiParser:
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
        try:
            self.truncate = self.dict.tokenizer.model_max_length
        except AttributeError:
            self.truncate = 100000
        self.doc_stride = opt['doc_stride']

        if shared:
            self.model = shared['model']
        else:
            self.model = self.build_model()
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
            loss = self.compute_loss(batch)
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
        n_best_size = 30
        max_answer_length = 50
        null_score_diff_threshold = 0.0
        tokenizer = self.dict.tokenizer
        if batch.batchsize <= 0:
            return
        else:
            bsz = batch.batchsize
        self.model.eval()
        loss, outputs = self.compute_loss(batch, return_output=True)
        no_answer_reply = batch.get('no_answer_reply', self.dict.cls_token)
        _PrelimPrediction = namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
        )
        all_predictions = OrderedDict()
        all_nbest_json = OrderedDict()
        scores_diff_json = OrderedDict()
        batch_best_preds = []
        for e_index, feature_indices in enumerate(batch.batch_indexes_map):
            prelim_predictions = []
            example = batch.examples[e_index]
            score_null = 1000000  # large and positive
            min_null_feature_index = 0  # the paragraph slice with min null score
            null_start_logit = 0  # the start logit at the slice with min null score
            null_end_logit = 0  # the end logit at the slice with min null score
            for f_index in feature_indices:
                feature = batch.features[f_index]
                f_output = [util.to_list(output[f_index]) for output in outputs.to_tuple() if len(output.size()) > 1]
                start_logits, end_logits = f_output
                result = SquadResult(f_index, start_logits, end_logits)
                start_indexes = _get_best_indexes(result.start_logits, n_best_size)
                end_indexes = _get_best_indexes(result.end_logits, n_best_size)
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = f_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # We could hypothetically create invalid predictions, e.g., predict
                        # that the start of the span is in the question. We throw out all
                        # invalid predictions.
                        if start_index >= len(feature.tokens):
                            continue
                        if end_index >= len(feature.tokens):
                            continue
                        if start_index not in feature.token_to_orig_map:
                            continue
                        if end_index not in feature.token_to_orig_map:
                            continue
                        if not feature.token_is_max_context.get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > max_answer_length:
                            continue
                        prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=f_index,
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=result.start_logits[start_index],
                                end_logit=result.end_logits[end_index],
                            )
                        )
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                )
            )
            prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
            _NbestPrediction = namedtuple(  # pylint: disable=invalid-name
                "NbestPrediction", ["text", "start_logit", "end_logit"]
            )
            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) >= n_best_size:
                    break
                feature = batch.features[pred.feature_index]
                if pred.start_index > 0:  # this is a non-null prediction
                    tok_tokens = feature.tokens[pred.start_index: (pred.end_index + 1)]
                    orig_doc_start = feature.token_to_orig_map[pred.start_index]
                    orig_doc_end = feature.token_to_orig_map[pred.end_index]
                    orig_tokens = example.doc_tokens[orig_doc_start: (orig_doc_end + 1)]

                    tok_text = tokenizer.convert_tokens_to_string(tok_tokens)
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = " ".join(orig_tokens)

                    final_text = get_final_text(tok_text, orig_text, True)
                    if final_text in seen_predictions:
                        continue

                    seen_predictions[final_text] = True
                else:
                    final_text = no_answer_reply
                    seen_predictions[final_text] = True

                nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit))

            # if we didn't include the empty option in the n-best, include it
            # if version_2_with_negative:
            if no_answer_reply not in seen_predictions:
                nbest.append(_NbestPrediction(text=no_answer_reply, start_logit=null_start_logit, end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text=no_answer_reply, start_logit=0.0, end_logit=0.0))

            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(_NbestPrediction(text=no_answer_reply, start_logit=0.0, end_logit=0.0))

            assert len(nbest) >= 1, "No valid predictions"

            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)
                if not best_non_null_entry:
                    if entry.text:
                        best_non_null_entry = entry

            probs = _compute_softmax(total_scores)

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                nbest_json.append(output)

            assert len(nbest_json) >= 1, "No valid predictions"

            # if not version_2_with_negative:
            #     all_predictions[example.qas_id] = nbest_json[0]["text"]
            # else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = no_answer_reply
                batch_best_preds.append(no_answer_reply)
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
                batch_best_preds.append(best_non_null_entry.text)
            all_nbest_json[example.qas_id] = nbest_json
        return Output(batch_best_preds)

    def compute_loss(self, batch, return_output=False):
        """
                Compute and return the loss for the given batch.

                Easily overridable for customized loss functions.

                If return_output is True, the full output from the call to self.model()
                is also returned, via a (loss, model_output) pair.
                """
        if batch.text_vec is None:
            raise ValueError('Cannot generate outputs without text vectors.')
        output = self.model(self._model_input(batch))
        batches_count = [1] * batch.batchsize
        self.record_local_metric('loss', AverageMetric.many([output['loss'].data.cpu()]*batch.batchsize, batches_count))

        if return_output:
            return output['loss'], output
        else:
            return output['loss']

    def _set_label_vec(self, obs, add_start, add_end, label_truncate):
        """
        Set the 'labels_vec' field in the observation.

        Useful to override to change vectorization behavior
        """
        # convert 'labels' or 'eval_labels' into vectors
        if not obs.get('features_vec', None):
            return None
        obs['label_vec'] = [f.input_ids[f.start_position:f.end_position+1] for f in obs['text_vec']]
        return obs

    def squad_convert_example_to_features(self, example, max_seq_length, doc_stride,
                                          max_query_length, padding_strategy, is_training):
        features = []
        tokenizer = self.dict.tokenizer
        if self.history_truncate > 0 and len(self.history.history_strings) > 1:
            tokens_count = 0
            history_text = ""
            for idx, dialogue in enumerate(reversed(self.history.history_dialogues)):
                if dialogue[1] is None:
                    continue
                d_text = "\n " + dialogue[0] + " " + dialogue[1]
                tokens = tokenizer.tokenize(d_text)
                tokens_count += len(tokens)
                if tokens_count <= self.history_truncate:
                    history_text += d_text
                else:
                    break
            truncated_history = tokenizer.encode(
                history_text, add_special_tokens=False, truncation=True, max_length=max_query_length)
            truncated_history.append(tokenizer.sep_token_id)
        else:
            truncated_history = []
        if is_training and not example.is_impossible:
            # Get start and end position
            start_position = example.start_position
            end_position = example.end_position

            # If the answer cannot be found in the text, then skip this example.
            actual_text = " ".join(example.doc_tokens[start_position: (end_position + 1)])
            cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
            if actual_text.find(cleaned_answer_text) == -1:
                logging.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                return []
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            if tokenizer.__class__.__name__ in [
                "RobertaTokenizer",
                "LongformerTokenizer",
                "BartTokenizer",
                "RobertaTokenizerFast",
                "LongformerTokenizerFast",
                "BartTokenizerFast",
            ]:
                sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
            else:
                sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1

            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
            )

        spans = []

        truncated_query = tokenizer.encode(
            example.question_text, add_special_tokens=False, truncation=True, max_length=max_query_length
        )

        # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
        # in the way they compute mask of added tokens.
        tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
        sequence_added_tokens = (
            tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
            if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
            else tokenizer.model_max_length - tokenizer.max_len_single_sentence
        )
        sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair
        span_doc_tokens = all_doc_tokens
        while len(spans) * doc_stride < len(all_doc_tokens):

            # Define the side we want to truncate / pad and the text/pair sorting
            if tokenizer.padding_side == "right":
                texts = truncated_query
                pairs = span_doc_tokens
                truncation = TruncationStrategy.ONLY_SECOND.value
            else:
                texts = span_doc_tokens
                pairs = truncated_query
                truncation = TruncationStrategy.ONLY_FIRST.value

            encoded_dict = tokenizer.encode_plus(
                texts,
                pairs,
                truncation=truncation,
                padding=padding_strategy,
                max_length=max_seq_length,
                return_overflowing_tokens=True,
                stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
                return_token_type_ids=True,
            )

            paragraph_len = min(
                len(all_doc_tokens) - len(spans) * doc_stride,
                max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
            )
            if len(truncated_history) > 0:
                if tokenizer.pad_token_id in encoded_dict["input_ids"]:
                    content_end_index = encoded_dict["input_ids"].index(tokenizer.pad_token_id) -1
                    history_end_index = min(content_end_index + len(truncated_history), max_seq_length)
                    encoded_dict["input_ids"][content_end_index:history_end_index] = truncated_history
                    encoded_dict["token_type_ids"][content_end_index:history_end_index] = torch.LongTensor(
                        [1] * len(truncated_history))
                    encoded_dict["attention_mask"][content_end_index:history_end_index] = torch.LongTensor(
                        [1] * len(truncated_history))
                else:
                    encoded_dict["input_ids"][-len(truncated_history):] = truncated_history

            if tokenizer.pad_token_id in encoded_dict["input_ids"]:
                if tokenizer.padding_side == "right":
                    non_padded_ids = encoded_dict["input_ids"][
                                     : encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
                else:
                    last_padding_id_position = (
                            len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(
                        tokenizer.pad_token_id)
                    )
                    non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1:]
            else:
                non_padded_ids = encoded_dict["input_ids"]

            tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

            token_to_orig_map = {}
            for i in range(paragraph_len):
                index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
                token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

            encoded_dict["paragraph_len"] = paragraph_len
            encoded_dict["tokens"] = tokens
            encoded_dict["token_to_orig_map"] = token_to_orig_map
            encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
            encoded_dict["truncated_history_with_special_tokens_length"] = len(truncated_history)
            encoded_dict["token_is_max_context"] = {}
            encoded_dict["start"] = len(spans) * doc_stride
            encoded_dict["length"] = paragraph_len

            spans.append(encoded_dict)

            if "overflowing_tokens" not in encoded_dict or (
                    "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
            ):
                break
            span_doc_tokens = encoded_dict["overflowing_tokens"]

        for doc_span_index in range(len(spans)):
            for j in range(spans[doc_span_index]["paragraph_len"]):
                is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
                index = (
                    j
                    if tokenizer.padding_side == "left"
                    else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
                )
                spans[doc_span_index]["token_is_max_context"][index] = is_max_context

        for span in spans:
            # Identify the position of the CLS token
            cls_index = span["input_ids"].index(tokenizer.cls_token_id)

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0)
            p_mask = np.ones_like(span["token_type_ids"])
            if tokenizer.padding_side == "right":
                p_mask[len(truncated_query) + sequence_added_tokens:] = 0
            else:
                p_mask[-len(span["tokens"]): -(len(truncated_query) + sequence_added_tokens)] = 0

            pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
            special_token_indices = np.asarray(
                tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
            ).nonzero()

            p_mask[pad_token_indices] = 1
            p_mask[special_token_indices] = 1

            # Set the cls index to 0: the CLS index can be used for impossible answers
            p_mask[cls_index] = 0

            span_is_impossible = example.is_impossible
            start_position = 0
            end_position = 0
            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = span["start"]
                doc_end = span["start"] + span["length"] - 1
                out_of_span = False

                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True

                if out_of_span:
                    start_position = cls_index
                    end_position = cls_index
                    span_is_impossible = True
                else:
                    if tokenizer.padding_side == "left":
                        doc_offset = 0
                    else:
                        doc_offset = len(truncated_query) + sequence_added_tokens

                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            features.append(
                SquadFeatures(
                    span["input_ids"],
                    span["attention_mask"],
                    span["token_type_ids"],
                    cls_index,
                    p_mask.tolist(),
                    example_index=0,
                    # Can not set unique_id and example_index here. They will be set after multiple processing.
                    unique_id=0,
                    paragraph_len=span["paragraph_len"],
                    token_is_max_context=span["token_is_max_context"],
                    tokens=span["tokens"],
                    token_to_orig_map=span["token_to_orig_map"],
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=span_is_impossible,
                    qas_id=example.qas_id,
                )
            )
        return features

    # Tokenize our training dataset
    def _set_text_vec(self, obs, history, truncate, is_training=True):
        # Tokenize contexts and questions (as pairs of inputs)
        if 'squad_example' not in obs:
            return obs
        features = self.squad_convert_example_to_features(
            example=obs['squad_example'],
            max_seq_length=self.truncate,
            doc_stride=self.doc_stride,
            max_query_length=self.query_truncate,
            padding_strategy="max_length",
            is_training=is_training
        )
        obs['text_vec'] = features
        labels_with_special_tokens = []
        for l in obs.get('labels', obs.get('eval_labels', [])):
            if l == "":
                labels_with_special_tokens.append(self.dict.cls_token)
            else:
                labels_with_special_tokens.append(l)
        if 'labels' in obs:
            obs.force_set('labels', labels_with_special_tokens)
        elif 'eval_labels' in obs:
            obs.force_set('eval_labels', labels_with_special_tokens)
        return obs


    def batchify(self, obs_batch, sort=False):
        is_training = self.is_training
        batch = Batch(batchsize=0)
        if len(obs_batch) == 0:
            return batch

        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if self.is_valid(ex)]

        if len(valid_obs) == 0:
            return batch
        valid_inds, exs = zip(*valid_obs)
        features = []
        batch_indexes_map = []
        examples = []
        unique_id = 1000000000
        cur_index = 0
        for example_index, example in valid_obs:
            cur_ex_docs = []
            squad_ex = example.get('squad_example')
            examples.append(squad_ex)
            example_features = example.get('text_vec', None)
            if not example_features:
                continue
            for example_feature in example_features:
                example_feature.example_index = example_index
                example_feature.unique_id = unique_id
                features.append(example_feature)
                cur_ex_docs.append(cur_index)
                unique_id += 1
                cur_index += 1
            example_index += 1
            batch_indexes_map.append(cur_ex_docs)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)

        all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)

        batch = Batch(
            batchsize=len(valid_obs),
            no_answer_reply=obs_batch[0].get('no_answer_reply', self.dict.cls_token),
            text_vec=all_input_ids,
            attention_mask=all_attention_masks,
            token_type_ids=all_token_type_ids,
            start_positions=all_start_positions,
            end_positions=all_end_positions,
            cls_index=all_cls_index,
            p_mask=all_p_mask,
            examples=examples,
            is_impossible=all_is_impossible,
            features_index=all_feature_index,
            batch_indexes_map=batch_indexes_map,
            valid_indices=valid_inds,
            features=features
        )
        if self.use_cuda:
            batch.text_vec = batch.text_vec.cuda()
            batch.attention_mask = batch.attention_mask.cuda()
            batch.token_type_ids = batch.token_type_ids.cuda()
            batch.start_positions = batch.start_positions.cuda()
            batch.end_positions = batch.end_positions.cuda()

        return batch

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
        inputs = {
            "input_ids": batch["text_vec"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch["token_type_ids"],
            "start_positions": batch["start_positions"],
            "end_positions": batch["end_positions"],
        }
        if self.model.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
            del inputs["token_type_ids"]
        if self.model.model_type in ["xlnet", "xlm"]:
            if self.use_cuda:
                inputs.update({"cls_index": batch["cls_index"].cuda(), "p_mask": batch["p_mask"].cuda(),
                               "is_impossible": batch["is_impossible"].cuda()})
            else:
                inputs.update({"cls_index": batch["cls_index"], "p_mask": batch["p_mask"],
                               "is_impossible": batch["is_impossible"]})
        return inputs

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

    def slice_text_with_token_index(self, text, start, length):
        if not text:
            return ""
        end = start+length
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

    # a shifting window size function that aims to find the start end indx for the max f1 score sequence in the context
    def get_token_start_end_position(self, context_encoding, answer_text, start_char, offset_allowance=50):
        answer_vector = self.dict.tokenizer.encode_plus(answer_text, add_special_tokens=False)
        normalised_answer_text = self.dict.tokenizer.decode(answer_vector['input_ids'])
        tokens_window_size = len(answer_vector['input_ids'])
        context_tokens = context_encoding['input_ids']
        max_no_context = len(context_tokens)
        best_answer_string = normalised_answer_text
        start = None
        end = None
        max_score = 0
        for i, _ in enumerate(context_tokens):
            cur_start = i
            cur_end = i + tokens_window_size
            if cur_end > max_no_context:
                break
            if cur_start == 0 and start_char != 0:
                continue
            else:
                char_count_before_start = len(self.dict.tokenizer.decode(context_tokens[:cur_start]))
                # Skip current step, because too early from the start char
                if char_count_before_start < start_char - offset_allowance:
                    continue
                # stop searching because oo far from the start char
                elif char_count_before_start > start_char + offset_allowance:
                    break
            cur_tokens = context_tokens[cur_start:cur_end]
            cur_string = self.dict.tokenizer.decode(cur_tokens)
            if cur_string == normalised_answer_text:
                start = cur_start
                end = cur_end
                best_answer_string = cur_string
                break
            else:
                _, _, cur_f1 = F1Metric._prec_recall_f1_score(cur_string, normalised_answer_text)
                if cur_f1 > max_score:
                    start = cur_start
                    end = cur_end
                    best_answer_string = cur_string
                    max_score = cur_f1
        return start, end, best_answer_string
