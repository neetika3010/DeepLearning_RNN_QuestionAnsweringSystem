# Text preprocessing and vocab building
def tokenize(text):
    text = text.lower()
    text = text.replace('?', '').replace(',', '')
    return text.split()

def build_vocab(df):
    vocab = {'<UNK>': 0}
    for _, row in df.iterrows():
        question_tokens = tokenize(row['question'])
        answer_tokens = tokenize(row['answer'])
        for token in question_tokens + answer_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
