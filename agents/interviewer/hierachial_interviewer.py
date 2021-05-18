import os

import torch
import torch.nn as nn
from parlai_internal.agents.interviewer.interviewer import InterviewerAgent, Sq2SqQuestionGenerationModel
from parlai_internal.utilities.flow_lstm_util import constants
from parlai_internal.utilities.flow_lstm_util.models.seq2seq import Seq2SeqModel

from parlai.core.torch_agent import Output
import parlai.utils.logging as logging
from parlai.core.opt import Opt


# class HierachialSeq2SeqModel(Seq2SeqModel):
#
#     def __init__(self, args, emb_matrix=None, use_cuda=False):
#         super.__init__(args, emb_matrix, use_cuda)
#         self.z_sent_size = 100
#         posterior_input_size = 0

#         self.softplus = nn.Softplus()
#         posterior_input_size = (config.num_layers
#                                 * config.encoder_hidden_size
#                                 * self.encoder.num_directions
#                                 + config.context_size)

#         self.prior_h = layers.FeedForward(self.hidden_dim,
#                                           self.hidden_dim,
#                                           num_layers=2,
#                                           hidden_size=config.context_size,
#                                           activation=config.activation)
#         self.prior_mu = nn.Linear(self.hidden_dim,
#                                   config.z_sent_size)
#         self.prior_var = nn.Linear(self.hidden_dim,
#                                    config.z_sent_size)
#
#         self.posterior_h = layers.FeedForward(posterior_input_size,
#                                               config.context_size,
#                                               num_layers=2,
#                                               hidden_size=config.context_size,
#                                               activation=config.activation)
#         self.posterior_mu = nn.Linear(config.context_size,
#                                       config.z_sent_size)
#         self.posterior_var = nn.Linear(config.context_size,
#                                        config.z_sent_size)
#         self.context2decoder = layers.FeedForward(config.context_size + config.z_sent_size,
#                                                   config.num_layers * config.decoder_hidden_size,
#                                                   num_layers=1,
#                                                   activation=config.activation)
#
#
#     def prior(self, context_outputs):
#         # Context dependent prior
#         h_prior = self.prior_h(context_outputs)
#         mu_prior = self.prior_mu(h_prior)
#         var_prior = self.softplus(self.prior_var(h_prior))
#         return mu_prior, var_prior
#
#     def posterior(self, context_outputs, encoder_hidden):
#         h_posterior = self.posterior_h(torch.cat([context_outputs, encoder_hidden], 1))
#         mu_posterior = self.posterior_mu(h_posterior)
#         var_posterior = self.softplus(self.posterior_var(h_posterior))
#         return mu_posterior, var_posterior




class HierachialSq2SqQuestionGenerationModel(Sq2SqQuestionGenerationModel):

    def __init__(self, opt, dict):
        # self.add_start_token = opt["add_start_token"]
        super().__init__(opt, dict)

        # init the model
        self.sq2sq_model = self.load_question_generation_model(opt, dict)
        self.config = self.sq2sq_model.args

    def load_question_generation_model(self, opt, dict):
        if opt.get('init_finetune', False):
            filename = os.path.join(opt['datapath'], constants.FINE_TUNE_FILE)
        else:
            filename = os.path.join(opt['datapath'], constants.BASE_MODEL_FILE)
        print(f"Loading model from '{filename}'...")
        checkpoint = torch.load(filename, lambda storage, loc: storage)
        args = checkpoint['config']
        if dict.vocab is not None:
            args['vocab'] = dict.vocab
        model = Seq2SeqModel(args, use_cuda=not opt['no_cuda'])
        model.load_state_dict(checkpoint['model'])
        return model


    def forward(self, xs, ys=None, prev_enc=None, maxlen=None, bsz=None):
        src = xs['src']
        src_mask = xs['src_mask']
        turn_ids = xs ['turn_ids']
        tgt_in = xs['tgt_in']
        bg = xs.get('bg', None)
        bg_mask = xs.get('bg_mask', None)
        # prepare for encoder/decoder
        B, T, L = tgt_in.size()
        tgt_in = tgt_in.view(B*T, L)
        dec_inputs = self.sq2sq_model.emb_drop(self.sq2sq_model.embedding(tgt_in))
        h_in, src_mask, (hn, cn), h_bg = self.sq2sq_model.encode_sources(src, src_mask, bg, bg_mask, turn_ids=turn_ids)
        log_probs, _, dec_attn, _ = self.sq2sq_model.decode(dec_inputs, hn, cn, h_in, src_mask, turn_ids=turn_ids, h_bg=self.sq2sq_model.drop(h_bg))
        _, preds = log_probs.max(dim=2)
        return log_probs[-1:, :, :], preds, {'h_in':h_in, 'src_mask': src_mask, 'hn': hn, 'cn': cn, 'h_bg': h_bg}


# class HierachialInterviewerAgent(InterviewerAgent):
#
#
#
