from src.data_loader import load_dataset
from src.preprocessing import build_vocab
from src.tokenize import encode_dataset
from src.model import RNNModel
from src.train import train

# 1. Load dataset
df = load_dataset("data/100_Unique_QA_Dataset.csv")

# 2. Build vocabulary
vocab = build_vocab(df)

# 3. Encode dataset to indices
encoded_data = encode_dataset(df, vocab)

# 4. Initialize model
model = RNNModel(vocab_size=len(vocab))

# 5. Train the model
train(model, encoded_data, vocab, epochs=10, lr=0.001)
