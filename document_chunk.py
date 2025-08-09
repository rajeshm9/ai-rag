

from langchain_text_splitters import RecursiveCharacterTextSplitter

def document_chunking(docs):
    """
    Splits documents into smaller chunks for better processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)
  
    return all_splits

