# Question Answering System using Recurrent Neural Networks (RNN)

This project implements a neural question-answering system using Recurrent Neural Networks (RNNs) in PyTorch. It is designed to demonstrate sequence modeling capabilities in Natural Language Processing (NLP) and forms part of my broader exploration into Generative AI and neural text understanding.

## Project Highlights

- Developed a **generative-style QA system** using a vanilla RNN architecture.
- Built a **custom preprocessing pipeline** for tokenization, lowercasing, punctuation handling, and dynamic vocabulary indexing.
- Implemented text encoding logic to convert natural language into padded index tensors using word-level tokenization.
- Integrated **embedding layers** and RNN cells (`nn.RNN`) to learn sequential dependencies between questions and answers.
- Trained the model over multiple epochs with **categorical cross-entropy loss** and **softmax-based prediction**, monitoring convergence using epoch-wise loss.
- Created a **modular codebase** using Python scripts to allow easy integration of LSTM/GRU, attention mechanisms, or pretrained embeddings in future iterations.

---

## Tools & Technologies

- **PyTorch** – Deep learning framework for model training and tensor operations
- **NLP** – Tokenization, vocabulary building, and sequence-to-sequence modeling
- **Python Modules** – `torch`, `pandas`, `numpy`
- **Core Concepts** – RNN, word embeddings, classification via softmax, vocabulary handling


## Tech Stack:
- Python 3.x
- PyTorch
- pandas, NumPy
- scikit-learn


## Folder Structure
 <pre> ``` RNN-QA-System/ ├── data/ │ └── 100_Unique_QA_Dataset.csv # Your question-answer dataset │ ├── src/ │ ├── data_loader.py # Load CSV data │ ├── preprocessing.py # Tokenization + vocab builder │ ├── tokenize.py # Index encoding for questions & answers │ ├── model.py # RNN model architecture │ └── train.py # Training loop │ ├── main.py # Calls and runs the whole training pipeline ├── requirements.txt ├── .gitignore └── README.md ``` </pre>

The project is structured into modular Python scripts and follows a clean pipeline:

1. Data Loading:Reads the QA dataset from a CSV
2. Text Preprocessing:Tokenizes and cleans questions and answers
3. Vocabulary Building: Builds a word-to-index mapping
4. Index Encoding:Converts text to sequences of indices
5. Modeling: Defines an RNN model in PyTorch
6. Training: Trains the model using the dataset


## How to run the project:

1.Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/RNN-QA-System.git
cd RNN-QA-System

2.Create a virtual environment
python -m venv venv
source venv/bin/activate  

3.Install dependencies
pip install -r requirements.txt

4.Run the training pipeline
python main.py

## Future Improvements

1.To add evaluation metrics (accuracy, F1, etc.)
2.To add LSTM/GRU-based model alternatives
3.Use pre-trained embeddings like GloVe
4.Implement an inference pipeline for testing new questions
5.Deploy using a Flask or FastAPI interface


