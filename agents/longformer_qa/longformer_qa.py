import os

import torch
import torch.nn as nn
from parlai_internal.agents.torch_span_agent.torch_span_agent import TorchSpanAgent, TorchExtractiveModel
from parlai_internal.utilities import util
from parlai.agents.hugging_face.dict import HuggingFaceDictionaryAgent
from parlai.utils.misc import warn_once
import parlai.utils.logging as logging
from parlai.utils.torch import padded_tensor
from transformers import LongformerModel, LongformerTokenizer, LongformerConfig, LongformerTokenizerFast
import collections

class LongformerEncoder(torch.nn.Module):
    """
        Bert Encoder.

        This encoder is initialized with the pretrained model from Hugging Face.
        """

    def __init__(self, opt, dict):
        super().__init__()
        self.transformer = self._init_from_pretrained(opt)
        self.use_cuda = not opt["no_cuda"] and torch.cuda.is_available()

    def _init_from_pretrained(self, opt):
        # load model
        # check if datapath has the files that hugging face code looks for
        if all(
            os.path.isfile(
                os.path.join(opt["datapath"], "models", "longformer_hf", file_name)
            )
            for file_name in ["pytorch_model.bin", "config.json"]
        ):
            fle_key = os.path.join(opt["datapath"], "models", "longformer_hf")
        else:
            fle_key = opt["longformer_type"]
        return LongformerModel.from_pretrained(fle_key)

class HFLongformerQAModel(TorchExtractiveModel):
    """
    """

    def __init__(self, opt, dict):

        # init the model
        super().__init__()
        self.encoder = LongformerEncoder(opt, dict)
        self.config = self.encoder.transformer.config
        self.hidden_size = self.config.hidden_size
        self.qa_outputs = nn.Linear(self.encoder.transformer.config.hidden_size, self.num_labels)

    def output(self, tensor):
        """
        Compute output logits.

        Because we concatenate the context with the labels using the
        `concat_without_padding` function, we must truncate the input tensor to return
        only the scores for the label tokens.
        """
        # get only scores for labels
        if self.text_lengths is not None:
            total_length = max(self.text_lengths)
            to_select = tensor.size(1) - total_length
            if not self.add_start_token:
                to_select = to_select + 1
            if to_select > 0:
                # select only label scores
                bsz = tensor.size(0)
                new_tensors = []
                for i in range(bsz):
                    start = self.text_lengths[i]
                    if not self.add_start_token:
                        start = start - 1
                    end = start + to_select
                    new_tensors.append(tensor[i : i + 1, start:end, :])
                tensor = torch.cat(new_tensors, 0)

        return self.lm_head(tensor)


class LongformerDictionaryAgent(HuggingFaceDictionaryAgent):
    def is_prebuilt(self):
        """
        Indicates whether the dictionary is fixed, and does not require building.
        """
        return True

    def get_tokenizer(self, opt):
        """
        Instantiate tokenizer.
        """
        fle_key = opt["longformer_type"]
        return LongformerTokenizerFast.from_pretrained(fle_key)

    def _define_special_tokens(self, opt):
        if opt["add_special_tokens"]:
            # Add addtional start/end/pad tokens
            self.unk_token = self.tokenizer.unk_token
            self.sep_token = self.tokenizer.sep_token
            self.pad_token = self.tokenizer.pad_token
            self.cls_token = self.tokenizer.cls_token
            self.mask_token = self.tokenizer.mask_token
            self.no_answer_token = self.tokenizer.unk_token

    def override_special_tokens(self, opt):
        # define special tokens
        self._define_special_tokens(opt)
        # now override
        self.unk_idx = self.tokenizer.convert_tokens_to_ids([self.unk_token])[0]
        self.sep_idx = self.tokenizer.convert_tokens_to_ids([self.sep_token])[0]
        self.pad_idx = self.tokenizer.convert_tokens_to_ids([self.pad_token])[0]
        self.cls_idx = self.tokenizer.convert_tokens_to_ids([self.cls_token])[0]
        self.mask_idx = self.tokenizer.convert_tokens_to_ids([self.mask_token])[0]
        # set tok2ind for special tokens
        self.tok2ind[self.unk_token] = self.unk_idx
        self.tok2ind[self.sep_token] = self.sep_idx
        self.tok2ind[self.pad_token] = self.pad_idx
        self.tok2ind[self.cls_token] = self.cls_idx
        self.tok2ind[self.mask_token] = self.mask_idx
        # set ind2tok for special tokens
        self.ind2tok[self.unk_idx] = self.unk_token
        self.ind2tok[self.sep_idx] = self.sep_token
        self.ind2tok[self.pad_idx] = self.pad_token
        self.ind2tok[self.cls_idx] = self.cls_token
        self.ind2tok[self.mask_idx] = self.mask_token


class LongformerQaAgent(TorchSpanAgent):
    """
    Hugging Face Bert Agent.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        super(LongformerQaAgent, cls).add_cmdline_args(argparser)
        agent = argparser.add_argument_group("LongformerQA Args")
        # query maximum length
        agent.add_argument(
            '--history-maximum-length',
            type=int,
            default=2046,
            help='maximum number of tokens allowed for history string',
        )
        # query maximum length
        agent.add_argument(
            '--query-maximum-length',
            type=int,
            default=512,
            help='maximum number of tokens allowed for quenry string',
        )
        # context maximum length
        agent.add_argument(
            '--context-maximum-length',
            type=int,
            default=1534,
            help='maximum number of tokens allowed for context string',
        )
        agent.add_argument(
            "--longformer-type",
            type=str,
            default="allenai/longformer-base-4096",
            choices=["allenai/longformer-base-4096", "allenai/longformer-large-4096"],
            help="Which size model to initialize.",
        )
        agent.add_argument(
            "--add-special-tokens",
            type="bool",
            default=True,
            help="Add special tokens (like PAD, etc.). If False, "
            "Can only use with batch size 1.",
        )
        argparser.set_defaults(
            text_truncate=4096,
            label_truncate=256,
            dict_maxexs=0,  # skip building dictionary
        )
        warn_once("WARNING: this model is in beta and the API is subject to change.")
        return agent

    def __init__(self, opt, shared=None):
        self.add_cls_token = opt.get('add_cls_token', False)
        self.sep_last_utt = opt.get('sep_last_utt', False)
        if not opt["add_special_tokens"] and opt["batchsize"] > 1:
            raise RuntimeError(
                "If using batchsize > 1, --add-special-tokens must be True."
            )
        super().__init__(opt, shared)
        self.query_truncate = opt['query_maximum_length']
        self.context_truncate = opt['context_maximum_length']
        self.history_truncate = opt['history_maximum_length']
        self.truncate = self.dict.tokenizer.max_len
        self.doc_stride = self.truncate

    @staticmethod
    def dictionary_class():
        """
        Return the dictionary class that this agent expects to use.

        Can be overriden if a more complex dictionary is required.
        """
        return LongformerDictionaryAgent

    def build_model(self, states=None):
        """
        Build and return model.
        """
        return HFLongformerQAModel(self.opt, self.dict)

    # Tokenize our training dataset
    def _set_text_vec(self, obs, history, truncate, is_training=True):
        # Tokenize contexts and questions (as pairs of inputs)
        if 'text' not in obs:
            return obs
        # The -3 accounts for [CLS], [SEP] and [SEP] and [SEP]

        start_positions = []
        end_positions = []
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = self.dict.tokenizer.tokenize(obs['context'])
        context_encodings = self.dict.tokenizer.encode_plus(obs['context'])
        ans_text = obs.get('single_label_text', None)
        history_text = self.truncate_with_dic(" ".join(history.history_strings[:-1]), self.history_truncate, latest=True)
        question_text = self.truncate_with_dic(obs['question_text'], self.query_truncate)
        query_tokens = self.dict.tokenizer.tokenize(question_text)
        # The -3 accounts for special tokens
        max_tokens_for_doc = self.truncate - len(query_tokens) - self.history_truncate - 4
        if is_training and obs['is_impossible']:
            tok_start_position = -1
            tok_end_position = -1
        elif is_training:
            start_idx, end_idx = self.get_correct_alignement(obs['context'], obs['single_label_text'],
                                                             int(obs['char_answer_start']))
            tok_start_position = context_encodings.char_to_token(start_idx)
            tok_end_position = context_encodings.char_to_token(end_idx-1)
        if tok_start_position is None or tok_end_position is None:
            print('no start')
        question_texts = []
        context_texts = []
        text_vecs = []

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


    def get_correct_alignement(self, context, gold_text, start_idx):
        """ Some original examples in SQuAD have indices wrong by 1 or 2 character. We test and fix this here. """
        end_idx = start_idx + len(gold_text)
        if context[start_idx:end_idx] == gold_text:
            return start_idx, end_idx  # When the gold label position is good
        elif context[start_idx - 1:end_idx - 1] == gold_text:
            return start_idx - 1, end_idx - 1  # When the gold label is off by one character
        elif context[start_idx - 2:end_idx - 2] == gold_text:
            return start_idx - 2, end_idx - 2  # When the gold label is off by two character
        else:
            raise ValueError()


    # Tokenize our training dataset
    def convert_to_features(self, example):
        # Tokenize contexts and questions (as pairs of inputs)
        input_pairs = [example['question'], example['context']]
        encodings = self.dict.tokenizer.encode_plus(input_pairs, pad_to_max_length=True, max_length=512)
        context_encodings = self.dict.tokenizer.encode_plus(example['context'])

        # Compute start and end tokens for labels using Transformers's fast tokenizers alignement methodes.
        # this will give us the position of answer span in the context text
        start_idx, end_idx = self.get_correct_alignement(example['context'], example['answers'])
        start_positions_context = context_encodings.char_to_token(start_idx)
        end_positions_context = context_encodings.char_to_token(end_idx - 1)

        # here we will compute the start and end position of the answer in the whole example
        # as the example is encoded like this <s> question</s></s> context</s>
        # and we know the postion of the answer in the context
        # we can just find out the index of the sep token and then add that to position + 1 (+1 because there are two sep tokens)
        # this will give us the position of the answer span in whole example
        sep_idx = encodings['input_ids'].index(self.tokenizer.sep_token_id)
        start_positions = start_positions_context + sep_idx + 1
        end_positions = end_positions_context + sep_idx + 1

        if end_positions > self.context_truncate:
            start_positions, end_positions = 0, 0

        encodings.update({'start_positions': start_positions,
                          'end_positions': end_positions,
                          'attention_mask': encodings['attention_mask']})
        return encodings