import PyPDF2
import os
import streamlit as st

import openai
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# environment variables
os.environ['OPENAI_API_KEY']=st.secrets["database"]["OPENAI_API_KEY"]
os.environ['LANGCHAIN_HANDLER']='langchain'

openai.api_key=os.environ['OPENAI_API_KEY']

def pdf_to_text(uploaded_file):
    pdfReader = PyPDF2.PdfReader(uploaded_file)
    no_pages = len(pdfReader.pages)
    text=""
    for page_index in range(no_pages):
        page = pdfReader.pages[page_index]
        text=text+page.extract_text()
    return text

def generate_summary(paper, prompt_template):
    # Define prompt
    prompt = PromptTemplate.from_template(prompt_template)

    # Define LLM chain
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="text"
    )

    # split the texts
    text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size = 15000,
            chunk_overlap  = 20,
            length_function = len,
            is_separator_regex = False,
        )

    final_texts = text_splitter.create_documents([paper])

    final_summary = []
    chain_steps = 2
    chain_run_time = int(len(final_texts)/chain_steps)
    if chain_run_time == 0:
        chain_run_time = 1
    print("Chain run time: ", chain_run_time)

    for index in range(0, len(final_texts), chain_run_time):
        response = stuff_chain.run(final_texts[index: index + chain_run_time])
        final_summary.append(response)

    summary = " ".join(final_summary)
    print("\nSummary: ", summary)

    return summary

def save_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100,length_function=len, is_separator_regex=False)
    docs = text_splitter.create_documents([text])

    embeddings = OpenAIEmbeddings()

    db = FAISS.from_documents(docs, embeddings)

    return db

def retrieval_augmented_generation(query, database):
  docs = database.similarity_search(query)
  chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
  return chain.run(input_documents=docs, question=query)
