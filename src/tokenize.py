from preprocessing import tokenize

def text_to_indices(text, vocab):
    return [vocab.get(token, vocab['<UNK>']) for token in tokenize(text)]

def encode_dataset(df, vocab):
    encoded = []
    for _, row in df.iterrows():
        q_idx = text_to_indices(row['question'], vocab)
        a_idx = text_to_indices(row['answer'], vocab)
        encoded.append((q_idx, a_idx))
    return encoded
