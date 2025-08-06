## RNN-Based Question Answering System (Script Version):
Built a question-answering system using RNNs in PyTorch.
Developed data preprocessing pipelines for tokenization and vocabulary handling.
Trained an RNN model with embeddings, focusing on multi-epoch loss tracking and softmax-based response prediction.

## Tech Stack:
- Python 3.x
- PyTorch
- pandas, NumPy


## Folder Structure

RNN-QA-System/
├── data/
│ └── 100_Unique_QA_Dataset.csv # Your question-answer dataset
│
├── src/
│ ├── data_loader.py # Load CSV data
│ ├── preprocessing.py # Tokenization + vocab builder
│ ├── tokenize.py # Index encoding for questions & answers
│ ├── model.py # RNN model architecture
│ └── train.py # Training loop
│
├── main.py # Calls and runs the whole training pipeline
├── requirements.txt
├── .gitignore
└── README.md

The project is structured into modular Python scripts and follows a clean pipeline:

1. Data Loading:Reads the QA dataset from a CSV
2. Text Preprocessing:Tokenizes and cleans questions and answers
3. Vocabulary Building: Builds a word-to-index mapping
4. Index Encoding:Converts text to sequences of indices
5.Modeling: Defines an RNN model in PyTorch
6. Training: Trains the model using the dataset


# #How to run the project:

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


