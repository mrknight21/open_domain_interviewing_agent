from collections import defaultdict
from parlai_internal.utilities.flow_lstm_util import constants
import numpy as np
import torch

def pad_char_start_end(char_ids):
    return [[constants.CHAR_START_ID] + w + [constants.CHAR_END_ID] for w in char_ids]

def pad_lists(x, fill_val=0, dtype=np.int64):
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


def map_data(tokenized_data, dict):
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
                  'context_char': map_char(item['context'], dict)}
        if item['qas']:
            keys_in_qas = item['qas'][0].keys()
        else:
            keys_in_qas = []
        retval['qas'] = []
        for x in item['qas']:
            qa = {}
            qa['question'] = map_field(x['question'], dict)
            qa['question_char'] = map_char(x['question'], dict)
            qa['question_idf'] = map_idf(x['question'], dict)
            qa['answer'] = map_field(x['answer'], dict)
            qa['answer_char'] = map_char(x['answer'], dict)
            qa['answer_idf'] = map_idf(x['answer'], dict)
            if 'start' in keys_in_qas:
                qa['start'] = x['start']
            if 'end' in keys_in_qas:
                qa['end'] = x['end']
            if 'followup' in keys_in_qas:
                qa['followup'] = x['followup']
            if 'yesno' in keys_in_qas:
                qa['yesno'] = x['yesno']
            retval['qas'].append(qa)
        return retval
    return map_one(tokenized_data, dict)


def generate_features(tok_data, idx_data, max_turns):
    tpara, ipara = tok_data, idx_data
    tsrc = [constants.TITLEST] + tpara['title'] + [constants.TITLEEN, constants.BGST] + tpara['background'][:constants.MAX_BACKGROUND] + [constants.BGEN]
    isrc = [constants.TITLEST_ID] + ipara['title'] + [constants.TITLEEN_ID, constants.BGST_ID] + ipara['background'][:constants.MAX_BACKGROUND] + [constants.BGEN_ID]
    csrc = pad_char_start_end([[constants.TITLEST_ID]] + ipara['title_char'] + [[constants.TITLEEN_ID],
           [constants.BGST_ID]] + ipara['background_char'][:constants.MAX_BACKGROUND] + [[constants.BGEN_ID]])

    tbg = [] + tsrc
    ibg = [] + isrc
    cbg = [] + csrc

    turn_ids = [-1] * len(tsrc)

    # clear src variables for dialogue history
    tsrc = []
    isrc = []
    csrc = []
    turn_ids = []

    tsrc += [constants.SECST] + tpara['section_title'] + [constants.SECEN]
    isrc += [constants.SECST_ID] + ipara['section_title'] + [constants.SECEN_ID]
    csrc += pad_char_start_end([[constants.SECST_ID]] + ipara['section_title_char'] + [[constants.SECEN_ID]])
    src_idf = [0] + ipara['section_title_idf'] + [0]
    turn_ids += [0] * (len(ipara['section_title']) + 2)
    qa_counts = len(tpara['qas'])
    if qa_counts > 0:
        keys_in_qas = tpara['qas'][0].keys()
    else:
        keys_in_qas = []
    ans_mask = np.zeros((len(ipara['context'][:constants.MAX_CONTEXT]), constants.MAX_TURNS), dtype=np.int64)

    this_para = defaultdict(list)
    for turnid, (tqa, iqa) in enumerate(zip(tpara['qas'], ipara['qas'])):
        ttgt_in = [constants.SOS] + tqa['question']
        ttgt_out = tqa['question'] + [constants.EOS]
        itgt_in = np.array([constants.SOS_ID] + iqa['question'], dtype=np.int64)
        itgt_out = np.array(iqa['question'] + [constants.EOS_ID], dtype=np.int64)
        ctgt_in = pad_char_start_end([[constants.SOS_ID]] + iqa['question_char'])
        ctgt_out = pad_char_start_end(iqa['question_char'] + [[constants.EOS_ID]])

        this_para['src_text'].append([] + tsrc)
        this_para['src_idx'].append(pad_lists(isrc, dtype=np.int64))
        this_para['src_char'].append(pad_lists(csrc, dtype=np.int64))
        this_para['src_idf'].append(pad_lists(src_idf, dtype=np.float32))
        this_para['tgt_text'].append([x.lower() for x in tqa['question']])
        this_para['tgt_in_idx'].append(pad_lists(itgt_in, dtype=np.int64))
        this_para['tgt_in_char'].append(pad_lists(ctgt_in, dtype=np.int64))
        this_para['tgt_out_idx'].append(pad_lists(itgt_out, dtype=np.int64))
        this_para['tgt_out_char'].append(pad_lists(ctgt_out, dtype=np.int64))
        this_para['turn_ids'].append(pad_lists(turn_ids, fill_val=-1, dtype=np.int64))
        if 'start' in keys_in_qas and 'end' in keys_in_qas:
            this_para['start'].append(iqa['start'])
            this_para['end'].append(iqa['end'])
            this_para['yesno'].append(iqa['yesno'])
            this_para['followup'].append(iqa['followup'])
            this_para['ans_mask'].append(np.array(ans_mask))
            ans_mask[iqa['start']:iqa['end'], turnid] += 1  # in span
            ans_mask[iqa['start']:iqa['start'] + 1, turnid] += 1  # beginning of span
            ans_mask[iqa['end'] - 1:iqa['end'], turnid] += 2  # end of span
        this_para['this_turnid'].append(turnid)
        # append Q and A with separators
        if turnid+1 < qa_counts:
            tsrc = [constants.QUESST] + [x.lower() for x in tqa['question']] + [constants.QUESEN, constants.ANSST] + \
                   tqa['answer'] + [constants.ANSEN]
            isrc = [constants.QUESST_ID] + iqa['question'] + [constants.QUESEN_ID, constants.ANSST_ID] + iqa[
                'answer'] + [constants.ANSEN_ID]
            csrc = pad_char_start_end(
                [[constants.QUESST_ID]] + iqa['question_char'] + [[constants.QUESEN_ID], [constants.ANSST_ID]] + iqa[
                    'answer_char'] + [[constants.ANSEN_ID]])
            turn_ids = [turnid + 1] * (len(tqa['question']) + len(tqa['answer']) + 4)
            ques_count = sum(1 for x in isrc if x == constants.QUESST_ID) - 1
            if max_turns >= 0 and ques_count > max_turns:
                idx = len(isrc) - 1
                count = 0
                while idx > 0:
                    if isrc[idx] == constants.QUESST_ID:
                        count += 1
                        if count > max_turns:
                            break
                    idx -= 1
                tsrc = tsrc[idx:]
                isrc = isrc[idx:]
                csrc = csrc[idx:]
                turn_ids = turn_ids[idx:]

    datum = dict()
    datum['ctx_text'] = tpara['context'][:constants.MAX_CONTEXT]
    datum['ctx_idx'] = ipara['context'][:constants.MAX_CONTEXT]
    datum['ctx_char'] = pad_lists(pad_char_start_end(ipara['context_char'][:constants.MAX_CONTEXT]), dtype=np.int64)

    datum['bg_text'] = tbg
    datum['bg_idx'] = ibg
    datum['bg_char'] = cbg
    datum['src_text'] = this_para['src_text']
    datum['tgt_text'] = this_para['tgt_text']

    for k in ['start', 'end', 'yesno', 'followup']:
        if k in keys_in_qas:
            datum[k] = this_para[k]

    for k in ['src_idx', 'src_char', 'tgt_in_idx', 'tgt_in_char', 'tgt_out_idx', 'tgt_out_char', 'ans_mask']:
        if k in this_para:
            datum[k] = pad_lists(this_para[k], dtype=np.int64)
    datum['this_turnid'] = this_para['this_turnid']
    datum['turn_ids'] = pad_lists(this_para['turn_ids'], fill_val=-1, dtype=np.int64)
    return datum





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


def prune_decoded_seqs(seqs):
    """
    Prune decoded sequences after EOS token.
    """
    out = []
    for s in seqs:
        if constants.EOS in s:
            idx = s.index(constants.EOS_TOKEN)
            out += [s[:idx]]
        else:
            out += [s]
    return out