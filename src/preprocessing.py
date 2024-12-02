"""
    Preprocessing Module.

    This module provides functions to extract text from PDFs, preprocess raw text
    and split it into manageable chunks for embedding generation.
"""
import re
import unicodedata
import fitz

def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file using PyMuPDF (as a fallback for complex layouts).
    """
    try:
        text = ""
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text("text")
        return text if text.strip() else "No text found in the document."
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def preprocess_text(text):
    """
    Cleans and preprocesses the extracted text.
    """
    try:
        # Normalize unicode characters
        text = unicodedata.normalize("NFKD", text)
        # Remove hyphenation (words split across lines)
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        # Fix OCR misrecognitions (example: converting "l" to "I")
        text = re.sub(r'\b(I|l)\b', 'I', text)
        # Replace non-breaking spaces (common in PDFs) with regular spaces
        text = text.replace("\xa0", " ")
        # Remove unwanted characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        # Return cleaned text
        return text.strip()
    except Exception as e:
        return f"Error in preprocessing: {str(e)}"


def split_text_into_chunks(text, chunk_size=500):
    """
        Splits text into chunks based on sentence boundaries,
        ensuring chunk sizes stay close to the limit.
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length <= chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            # Add chunk if it's non-empty
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length

    # Add the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks
