import os
import torch
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
from docx import Document
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

class LegalDocumentSummarizer:
    def __init__(self):
        # Initialize the model and tokenizer
        self.model_name = "facebook/bart-large-cnn"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        # Download required NLTK data
        nltk.download('punkt')
        
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_sm")
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def preprocess_text(self, text):
        """Preprocess the legal text by removing unnecessary whitespace and normalizing."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def chunk_text(self, text, max_length=1024):
        """Split text into chunks that fit within model's maximum length."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(self.tokenizer.encode(sentence))
            if current_length + sentence_length > max_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def summarize_chunk(self, text, max_length=150, min_length=50):
        """Summarize a single chunk of text."""
        inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        inputs = inputs.to(self.device)

        summary_ids = self.model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def summarize_document(self, text, max_length=150, min_length=50):
        """Summarize an entire document by processing it in chunks."""
        # Preprocess the text
        text = self.preprocess_text(text)
        
        # Split into chunks
        chunks = self.chunk_text(text)
        
        # Summarize each chunk
        summaries = []
        for chunk in tqdm(chunks, desc="Summarizing chunks"):
            summary = self.summarize_chunk(chunk, max_length, min_length)
            summaries.append(summary)
        
        # Combine summaries
        final_summary = ' '.join(summaries)
        
        # If the final summary is too long, summarize it again
        if len(self.tokenizer.encode(final_summary)) > 1024:
            final_summary = self.summarize_chunk(final_summary, max_length, min_length)
        
        return final_summary

    def process_docx_file(self, file_path):
        """Process a .docx file and return its text content."""
        doc = Document(file_path)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return '\n'.join(text)

def main():
    # Initialize the summarizer
    summarizer = LegalDocumentSummarizer()
    
    print("Legal Document Summarizer")
    print("------------------------")
    
    while True:
        print("\nOptions:")
        print("1. Enter text directly")
        print("2. Process a .docx file")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            text = input("\nEnter the legal text to summarize:\n")
            summary = summarizer.summarize_document(text)
            print("\nSummary:")
            print(summary)
            
        elif choice == "2":
            file_path = input("\nEnter the path to the .docx file: ")
            try:
                text = summarizer.process_docx_file(file_path)
                summary = summarizer.summarize_document(text)
                print("\nSummary:")
                print(summary)
            except Exception as e:
                print(f"Error processing file: {str(e)}")
                
        elif choice == "3":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 