# Legal Document Summarizer

This program provides an automated solution for summarizing legal documents using advanced Natural Language Processing (NLP) techniques. It uses the BART model, which is particularly well-suited for summarization tasks.

## Features

- Summarize legal text directly input by the user
- Process and summarize .docx files
- Handles long documents by chunking and summarizing in parts
- Uses state-of-the-art transformer models for high-quality summaries
- Progress bar to track summarization progress
- GPU acceleration when available

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Download the spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Run the program:
```bash
python legal_summarizer.py
```

2. Choose from the following options:
   - Option 1: Enter legal text directly
   - Option 2: Process a .docx file
   - Option 3: Exit the program

3. For .docx files, provide the full path to the file when prompted.

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- spaCy
- python-docx
- NLTK
- tqdm

## Notes

- The program uses the BART-large-CNN model, which is optimized for summarization tasks
- Long documents are automatically split into chunks and processed
- The summary length can be adjusted by modifying the `max_length` and `min_length` parameters in the code
- GPU acceleration is automatically enabled if available

## Limitations

- The model has a maximum input length of 1024 tokens
- Very long documents may take some time to process
- The quality of summaries may vary depending on the complexity and structure of the legal text 