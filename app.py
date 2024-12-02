"""
    Document Query Bot Application.

    This Streamlit-based application allows users to upload PDF documents,
    extract and preprocess text, generate embeddings, and query the document
    for relevant information. It integrates several modular components for preprocessing,
    embeddings generation, and vector database management.

    Modules:
    - `src.embeddings`: Handles embedding creation and querying.
    - `src.preprocessing`: Provides utilities for text extraction and preprocessing.
    - `src.vector_db`: Manages the vector database for storing and searching embeddings.
"""

import os
import streamlit as st
from src.embeddings import create_embeddings, query_embeddings
from src.preprocessing import extract_text_from_pdf, preprocess_text, split_text_into_chunks
from src.vector_db import load_vector_db, save_vector_db

# Paths
DOCUMENTS_DIR = "documents/"
DB_PATH = "data/vector_db.pkl"

# Ensure directories exist
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Streamlit UI
st.title("Document Query Bot")
st.sidebar.header("Upload and Query")

# Initialize a flag to check if embeddings have already been created in session state
if 'embeddings_created' not in st.session_state:
    st.session_state.embeddings_created = False

# Upload PDF
uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    """
        Handles file upload and text extraction. 
        Generates embeddings and updates the vector database 
        if embeddings are not yet created in the session.
    """
    file_path = os.path.join(DOCUMENTS_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.sidebar.success(f"Uploaded {uploaded_file.name}")

    # Extract text
    with st.spinner("Extracting text from PDF..."):
        raw_text = extract_text_from_pdf(file_path)
        cleaned_text = preprocess_text(raw_text)
        chunks = split_text_into_chunks(cleaned_text)

    # Create embeddings and store in vector database
    if not st.session_state.embeddings_created:  # Only create embeddings if they haven't been created
        with st.spinner("Creating embeddings and updating the database..."):
            embeddings = create_embeddings(chunks)
            vector_db = load_vector_db(DB_PATH)
            vector_db.add(embeddings)
            save_vector_db(vector_db, DB_PATH)
            st.sidebar.success("Embeddings created and stored!")
            st.session_state.embeddings_created = True  # Set the flag to True after embeddings are created

# Query Section
st.header("Ask Your Document")
query = st.text_input("Enter your query:")

if query.strip():
    """
        Handles querying the document embeddings. 
        Displays the top result along with its similarity score.
    """
    if st.session_state.embeddings_created:
        with st.spinner("Fetching results..."):
            results = query_embeddings(query, DB_PATH)
            if results:
                st.subheader("Top Query Result")
                # Only show the top result
                top_result = results[0]  # Get the first result from the list
                st.markdown(f"**Result:** {top_result[0]}")  # Display the text of the top result
                st.markdown(f"**Similarity Score:** {top_result[1]:.2f}")  # Display the score of the top result
            else:
                st.warning("No relevant results found. Try rephrasing your query.")
    else:
        st.warning("Please upload a document and create embeddings first.")
else:
    st.warning("Please enter a valid query.")
