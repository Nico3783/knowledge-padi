import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document

# Default text splitter parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def parse_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        str: Extracted text from the PDF.
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"Error reading PDF file {pdf_path}: {e}")


def preprocess_pdf(pdf_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Preprocesses a single PDF file into text chunks.
    
    Args:
        pdf_path (str): Path to the PDF file.
        chunk_size (int): Maximum size of each text chunk.
        chunk_overlap (int): Overlap size between chunks.
    
    Returns:
        list: List of text chunks.
    """
    text = parse_pdf(pdf_path)
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    print(f"Processed {pdf_path}: Extracted {len(chunks)} chunks.")
    return chunks


def preprocess_texts(directory_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Preprocesses and chunks text from all PDFs in the given directory.
    
    Args:
        directory_path (str): Path to the directory containing PDF files.
        chunk_size (int): Maximum size of each text chunk.
        chunk_overlap (int): Overlap size between chunks.
    
    Returns:
        list: List of Document objects, each with a unique id.
    """
    # Validate directory
    if not os.path.isdir(directory_path):
        raise RuntimeError(f"The directory '{directory_path}' does not exist or is not accessible.")

    all_documents = []
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Process each file in the directory
    for filename in os.listdir(directory_path):
        pdf_path = os.path.join(directory_path, filename)

        if not filename.endswith(".pdf"):
            print(f"Skipping non-PDF file: {filename}")
            continue

        if not os.path.isfile(pdf_path):
            print(f"Skipping directory or invalid file: {filename}")
            continue

        try:
            text = parse_pdf(pdf_path)
            chunks = text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                doc_id = f"{filename}_{i}"
                document = Document(page_content=chunk, id=doc_id)
                all_documents.append(document)
            print(f"Processed '{filename}': {len(chunks)} chunks extracted.")
        except Exception as e:
            print(f"Error processing '{filename}': {e}")

    print(f"Completed processing. Total documents created: {len(all_documents)}")
    return all_documents

