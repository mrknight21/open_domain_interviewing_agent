from parlai_internal.utilities.flow_lstm_util import constants
import numpy as np
import torch

def pad_char_start_end(char_ids):
    return [[constants.CHAR_START_ID] + w + [constants.CHAR_END_ID] for w in char_ids]

def pad_lists(self, x, fill_val=0, dtype=np.int64):
    size = [len(x)]
    y = x
    while isinstance(y[0], list) or isinstance(y[0], np.ndarray):
        yy = []
        mx = 0
        for t in y:
            mx = max(len(t), mx)
            yy.extend(t)
        size.append(mx)
        y = yy

    res = np.full(size, fill_val, dtype=dtype)
    assert len(size) <= 4
    if len(size) == 1:
        res = np.array(x, dtype=dtype)
    elif len(size) == 2:
        for i in range(len(x)):
            res[i, :len(x[i])] = x[i]
    elif len(size) == 3:
        for i in range(len(x)):
            for j in range(len(x[i])):
                res[i, j, :len(x[i][j])] = x[i][j]
    elif len(size) == 4:
        for i in range(len(x)):
            for j in range(len(x[i])):
                for k in range(len(x[i][j])):
                    res[i, j, k, :len(x[i][j][k])] = x[i][j][k]

    return res

def map_data(self, tokenized_data, dict):
    def map_field(field, dict):
        return [dict.tok2ind.get(x.lower(), dict.unk_idx) for x in field]

    def map_char(field, dict, do_lower=False):
        if do_lower:
            return [[dict.char2id.get(c, dict.unk_idx) for c in w.lower()] if w not in constants.VOCAB_PREFIX else [
                dict.char2id[w]] for w in field]
        else:
            return [
                [dict.char2id.get(c, dict.unk_idx) for c in w] if w not in constants.VOCAB_PREFIX else [dict.char2id[w]]
                for w in field]

    def map_idf(field, dict):
        return [1 / dict.wordid2docfreq[dict.tok2ind.get(x.lower(), dict.unk_idx)] for x in field]

    def copy_mask(src, dst, dcit):
        return [[1 if w1.lower() == w2.lower() else 0 for w2 in dst] for w1 in src]

    def map_one(item, dict):
        retval = {'title': map_field(item['title'], dict),
                  'title_char': map_char(item['title'], dict),
                  'section_title': map_field(item['section_title'], dict),
                  'section_title_idf': map_idf(item['section_title'], dict),
                  'section_title_char': map_char(item['section_title'], dict),
                  'background': map_field(item['background'], dict),
                  'background_char': map_char(item['background'], dict),
                  'context': map_field(item['context'], dict),
                  'context_char': map_char(item['context'], dict),
                  'qas': [{'question': map_field(x['question'], dict), 'answer': map_field(x['answer'], dict),
                           'question_char': map_char(x['question'], dict, do_lower=True),
                           'answer_char': map_char(x['answer'], dict),
                           'question_idf': map_idf(x['question'], dict), 'answer_idf': map_idf(x['answer'], dict),
                           'start': x['start'],
                           'end': x['end'],
                           'followup': x['followup'],
                           'yesno': x['yesno']} for x in item['qas']]}
        return retval
    return map_one(tokenized_data, dict)


def _collate_fn(self, obs_batch):
    batch_data = [x['text_vec'] for x in obs_batch]
    src = torch.from_numpy(pad_lists([x['src_idx'] for x in batch_data]))
    src_char = torch.from_numpy(pad_lists([x['src_char'] for x in batch_data]))
    ctx = torch.from_numpy(pad_lists([x['ctx_idx'] for x in batch_data]))
    ctx_char = torch.from_numpy(pad_lists([x['ctx_char'] for x in batch_data]))
    tgt_in = torch.from_numpy(pad_lists([x['tgt_in_idx'] for x in batch_data]))
    tgt_in_char = torch.from_numpy(pad_lists([x['tgt_in_char'] for x in batch_data]))
    tgt_out = torch.from_numpy(pad_lists([x['tgt_out_idx'] for x in batch_data]))
    tgt_out_char = torch.from_numpy(pad_lists([x['tgt_out_char'] for x in batch_data]))
    # neg_out = torch.from_numpy(self.pad_lists([x['neg_out_idx'] for x in batch_data]))
    # neg_out_char = torch.from_numpy(self.pad_lists([x['neg_out_char'] for x in batch_data]))

    this_turn = torch.from_numpy(pad_lists([x['this_turnid'] for x in batch_data]))
    ans_mask = torch.from_numpy(pad_lists([x['ans_mask'] for x in batch_data]))

    turn_ids = torch.from_numpy(pad_lists([x['turn_ids'] for x in batch_data], fill_val=-1))
    start = torch.from_numpy(pad_lists([[x['start'][-1]] for x in batch_data], fill_val=-1))
    end = torch.from_numpy(pad_lists([[x['end'][-1]] for x in batch_data], fill_val=-1))
    yesno = torch.from_numpy(pad_lists([[x['yesno'][-1]] for x in batch_data], fill_val=-1))
    followup = torch.from_numpy(pad_lists([[x['followup'][-1]] for x in batch_data], fill_val=-1))

    retval = {'src': src,
              'src_char': src_char,
              'src_text': [x['src_text'] for x in batch_data],
              'tgt_in': tgt_in,
              'tgt_out': tgt_out,
              'tgt_out_char': tgt_out_char,
              'tgt_text': [x['tgt_text'] for x in batch_data],
              'turn_ids': turn_ids,
              'ctx': ctx,
              'ctx_char': ctx_char,
              'ctx_text': [x['ctx_text'] for x in batch_data],
              'start': start,
              'end': end,
              'yesno': yesno,
              'followup': followup,
              'this_turn': this_turn,
              'ans_mask': ans_mask}
    if 'bg_text' in batch_data[0]:
        bg = torch.from_numpy(pad_lists([x['bg_idx'] for x in batch_data]))
        bg_char = torch.from_numpy(pad_lists([x['bg_char'] for x in batch_data]))
        retval['bg'] = bg
        retval['bg_char'] = bg_char
        retval['bg_text'] = [x['bg_text'] for x in batch_data]
    return retval

