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
