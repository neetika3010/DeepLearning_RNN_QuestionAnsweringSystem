import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train(model, dataset, vocab, epochs=10, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for question_idx, answer_idx in dataset:
            input_tensor = torch.tensor([question_idx], dtype=torch.long)
            target_tensor = torch.tensor(answer_idx, dtype=torch.long)

            optimizer.zero_grad()
            output = model(input_tensor)
            loss = loss_fn(output, target_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")
 