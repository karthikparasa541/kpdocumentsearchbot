#This code is written to create a PDF reading chatbot using RAG techniques

#installing all the required libraries
%pip install langchain-openai
%pip install langchain
%pip install streamlit
%pip install openai
%pip install PyPDF2
%pip install langchain-community
%pip install faiss-cpu
%pip install openai
%pip install tiktoken
%pip install python-dotenv
%pip install langchain-classic

# saving the api key

import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()



# --- Streamlit UI ---

def main():

st.header("Multi Modal Chat using RAG📚")
st.subheader("Chat using a PDF or Website")

web = st.text_input("Enter the Website to read the content:")

pdf_docs = st.file_uploader(
            "Upload your PDFs here",
            type="pdf",
            accept_multiple_files=True
        )

query = st.text_input("Ask questions about your documents:")

bt1, bt2, bt3 = st.columns(3)

with bt1:
  if st.button("Search in Document"):
    #invoking the LLM model with the prompt
    st.write("***** Searching in the Document ************")
    
with bt2:
  if st.button("Search in Website"):
    #invoking the LLM model with the prompt
    st.write("***** Searching in the Website ************")

with bt3:
  if st.button("Clear"):
    st.write("")