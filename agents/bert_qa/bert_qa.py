import os

import torch
import torch.nn as nn
from parlai_internal.agents.torch_span_agent.torch_span_agent import TorchSpanAgent, TorchExtractiveModel
from parlai.agents.hugging_face.dict import HuggingFaceDictionaryAgent
from parlai.utils.misc import warn_once
import parlai.utils.logging as logging
from parlai.utils.torch import padded_tensor
from transformers import BertTokenizer, BertModel

SPECIAL_TOKENS = {"unk_token": "[UNK]", "sep_token": "[SEP]",
                  "pad_token": "[PAD]", "cls_token": "[CLS]",
                  "mask_token": "[MASK]"}

class BertEncoder(torch.nn.Module):
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
                os.path.join(opt["datapath"], "models", "bert_hf", file_name)
            )
            for file_name in ["pytorch_model.bin", "config.json"]
        ):
            fle_key = os.path.join(opt["datapath"], "models", "bert_hf")
        else:
            fle_key = opt["bert_type"]
        return BertModel.from_pretrained(fle_key)

class HFBertQAModel(TorchExtractiveModel):
    """
    """

    def __init__(self, opt, dict):

        # init the model
        super().__init__()
        self.encoder = BertEncoder(opt, dict)
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


class BertDictionaryAgent(HuggingFaceDictionaryAgent):
    def is_prebuilt(self):
        """
        Indicates whether the dictionary is fixed, and does not require building.
        """
        return True

    def get_tokenizer(self, opt):
        """
        Instantiate tokenizer.
        """
        fle_key = opt["bert_type"]
        return BertTokenizer.from_pretrained(fle_key)

    def _define_special_tokens(self, opt):
        if opt["add_special_tokens"]:
            # Add addtional start/end/pad tokens
            self.tokenizer.add_special_tokens(SPECIAL_TOKENS)
            self.unk_token = SPECIAL_TOKENS["unk_token"]
            self.sep_token = SPECIAL_TOKENS["sep_token"]
            self.pad_token = SPECIAL_TOKENS["pad_token"]
            self.cls_token = SPECIAL_TOKENS["cls_token"]
            self.mask_token = SPECIAL_TOKENS["mask_token"]
            self.no_answer_token = SPECIAL_TOKENS["cls_token"]

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


class BertQuestionAnsweringAgent(TorchSpanAgent):
    """
    Hugging Face Bert Agent.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        agent = argparser.add_argument_group("BertQa Args")
        agent.add_argument(
            "--bert-type",
            type=str,
            default="bert-base-uncased",
            choices=["bert-base-uncased", "bert-base-cased", "distilbert-base-uncased", "distilbert-base-cased", "distilgpt2"],
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
            text_truncate=768,
            label_truncate=256,
            dict_maxexs=0,  # skip building dictionary
        )
        super(BertQuestionAnsweringAgent, cls).add_cmdline_args(argparser)
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

    @staticmethod
    def dictionary_class():
        """
        Return the dictionary class that this agent expects to use.

        Can be overriden if a more complex dictionary is required.
        """
        return BertDictionaryAgent

    def build_model(self, states=None):
        """
        Build and return model.
        """
        return HFBertQAModel(self.opt, self.dict)




