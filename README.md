# **Document Query Bot**

The **Document Query Bot** is a user-friendly application that enables you to upload PDF documents, preprocess and analyze the text using advanced machine learning techniques, and query the document content for relevant information. It combines a powerful vector search engine with embeddings generated using a pre-trained transformer model to provide precise and relevant responses.

---

## **Features**

- **PDF Upload:** Easily upload your PDF documents for processing.
- **Text Extraction:** Extracts text from PDF files using PyMuPDF.
- **Text Preprocessing:** Cleans and prepares the extracted text for analysis.
- **Embeddings Creation:** Generates embeddings using the `all-MiniLM-L6-v2` model from `SentenceTransformers`.
- **Query Engine:** Allows users to input a query and retrieves the most relevant content from the uploaded documents.
- **Persistent Storage:** Stores embeddings in a FAISS-based vector database for quick and efficient retrieval.
- **Interactive UI:** Built using Streamlit, providing a seamless experience for uploading documents, creating embeddings, and querying.

---

## **Getting Started**

Follow these steps to set up and run the application.

### **1. Prerequisites**

Ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment tool (optional, e.g., `venv` or `conda`)

### **2. Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/jforjatin/document-query-bot.git
   cd document-query-bot

2. Install dependencies:

    ```bash
    pip install -r requirements.txt

3. Create necessary directories:

    ```bash
    mkdir documents data model

4. Download the model files: 
   
   Go to this link(https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/all-MiniLM-L6-v2.zip), download the provided zip file, and extract its contents into the model folder.
   
### **3. Running the Application**

1. Start the application by running:

    ```bash
    streamlit run app.py
