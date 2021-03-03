def get_start_end_idx(gold_text, start_idx, context):
    end_idx = start_idx + len(gold_text)
    # sometimes squad answers are off by a character or two – fix this
    if context[start_idx:end_idx] == gold_text:
        return start_idx, end_idx
    elif context[start_idx - 1:end_idx - 1] == gold_text:
        start_idx = start_idx - 1
        end_idx = end_idx - 1  # When the gold label is off by one character
    elif context[start_idx - 2:end_idx - 2] == gold_text:
        start_idx = start_idx - 2
        end_idx = end_idx - 2  # When the gold label is off by two characters
    return start_idx, end_idx


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def build_char_word_offset_list(paragraph_text):
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in paragraph_text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)
    return doc_tokens, char_to_word_offset


def piece_word_char_to_token(tokens, start_index, answer):
    token_start_index = []
    for index, ans in zip(start_index, answer):
        char_index = 0
        for token in tokens:
            subword = False
            if '##' in token:
                subword = True
                word = token.replace('##', '')


def _check_is_max_context(doc_spans, cur_span_index, position):
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


def improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))
    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end
    return input_start, input_end

def get_correct_alignement(context, gold_text, start_idx):
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


def to_list(tensor):
    return tensor.detach().cpu().tolist()

