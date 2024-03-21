import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

from langchain.document_loaders import TextLoader
loader = TextLoader('./Ria_Banerjee_resume.txt')
documents = loader.load()

import textwrap

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=2)
docs = text_splitter.split_documents(documents)

from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()

from langchain.vectorstores import FAISS

db = FAISS.from_documents(docs, embeddings)

while True:
    print("Enter prompt: ")
    query = input()
    docs = db.similarity_search(query)
    print(wrap_text_preserve_newlines(str(docs[0].page_content)))
