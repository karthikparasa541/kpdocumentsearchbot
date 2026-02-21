#This code is written to create a PDF reading chatbot using RAG techniques


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
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import RetrievalQA

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

#Custom functions
def get_pdf_text(pdf_docs):
    """Extract text from a list of PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_conversation_chain(vectorstore):
    """Create a conversational retrieval chain."""
    llm = ChatOpenAI(temperature=0.7, model_name='gpt-4o')
    
    template = """You are a helpful AI assistant for querying PDF documents.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer concise and relevant.
    
    Context: {context}
    Question: {question}
    Helpful Answer:"""

    prompt = PromptTemplate(input_variables=['context', 'question'], template=template)
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        combine_docs_chain_kwargs={'prompt': prompt}
    )



def process_documents(pdf_docs, query):
    """Process uploaded PDF documents."""
    try:
        raw_text = get_pdf_text(pdf_docs)
        if not raw_text.strip():
            st.warning("Could not extract text from the PDF(s). They might be image-based or empty.")
            return

        text_chunks = get_text_chunks(raw_text)
        embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        llm = ChatOpenAI(temperature=0.7, model_name='gpt-4o-mini')
        qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
        )
        
        response = ask_question(query)
        return response                
        
    except Exception as e:
        st.error(f"An error occurred: {e}")




# --- Streamlit UI ---

def main():

            st.header("Multi Modal Chat using RAG📚")
            st.subheader("Chat using a PDF or Website")
            api_key_loaded = os.getenv("OPENAI_API_KEY") is not None
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
                        if not pdf_docs:
                            st.error("Please upload at least one PDF file.")
                #invoking the LLM model with the prompt
                        else:
                                    st.write("***** Searching in the Document ************")
                                    answer = process_documents(pdf_docs, query)
                                    st.write(answer)
                                    
                
            with bt2:
              if st.button("Search in Website"):
                #invoking the LLM model with the prompt
                st.write("***** Searching in the Website ************")
            
            with bt3:
              if st.button("Clear"):
            
                st.write("")

if __name__ == '__main__':
    main()



















