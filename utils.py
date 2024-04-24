import pandas as pd
import numpy as np
import os
from io import StringIO 
import json
from enum import Enum

from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec, PodSpec

def reset_pinecone_db():
    """
    Resets the starter Pinecone database and initializes the testing index named 'test'
    This is necessary as starter Pinecone account only allows for a single index
    """

    print("Resetting Pinecone database")

    pinecone = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    INDEX_NAME = 'test'
    if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
      pinecone.delete_index(INDEX_NAME)

    pinecone.create_index(name=INDEX_NAME, 
    dimension=1536,  #corresponds to the embedding size of OpenAI embeddings
    metric='cosine',
    spec = PodSpec(environment="gcp-starter")
    )


def retrieve_pdf_docs(folder):
    """
    Retrieves the raw text from pdf documents listed within the folder
    """
    
    docs=[]
    for file in os.listdir(folder):
        loader = PyPDFLoader(os.path.join(folder,file))
        docs += loader.load()
        
    return docs


def format_docs(docs):
    return "\n\n ------------".join(doc.page_content for doc in docs)


def get_qa_prompt():
    """
    Returns the Langchain prompt for running the Question-Answer interaction
    """

    qa_template = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    Question: {question} 
    Context: {context} 
    Answer:
    """
    qa_prompt = ChatPromptTemplate.from_template(qa_template)

    return qa_prompt


def get_retriever(docs, database, k):
    """
    Returns a retriever based on the database type specified. Can select from one of 3 options : Faiss, Chroma, Pinecone
    """

    if database == 'faiss':
        store = FAISS.from_documents(docs, OpenAIEmbeddings(), distance_strategy=DistanceStrategy.COSINE)
    elif database == 'chroma':
        store =  Chroma.from_documents(docs, OpenAIEmbeddings(), collection_metadata={"hnsw:space": "cosine"})
    elif database == 'pinecone':
        reset_pinecone_db()
        store = PineconeVectorStore.from_documents(docs, OpenAIEmbeddings(), index_name='test')  #pinecone uses cosine similarity by default
    else:
        raise ValueError('database value should be one of the following : faiss, chroma, pinecone')

    
    retriever = store.as_retriever(search_kwargs={"k": k})

    return retriever



def get_llm_eval_chain():
    """
    Returns a Langchain pipeline for running a LLM self eval script 

    """


    class gradeEnum(str,Enum):
        correct = "correct"
        incorrect = "incorrect"
        
    class LLMEvalResult(BaseModel):
        grade: gradeEnum = Field(description="Final grade label. Accepted labels : Correct, Incorrect")
        description: str = Field(description="Explanation of why the specific grade was assigned. Must be concise. Not more than 2 sentences")
    
    json_parser = JsonOutputParser(pydantic_object=LLMEvalResult)
    
    qa_eval_prompt_text = """
    You are a teacher evaluating a test. 
    You are provided with a question along with an answer for the question written by a student. Evaluate the question-answer pair and provide feedback.
    {format_instructions}
    Question : {question}
    Answer : {answer}
    """
    
    qa_eval_prompt = PromptTemplate(
        template=qa_eval_prompt_text,
        input_variables=["question","answer"],
        partial_variables={"format_instructions": json_parser.get_format_instructions()},
    )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    llm_eval_chain = qa_eval_prompt | llm | json_parser
    
    return llm_eval_chain


def displayResult(x):
    """
    Helper function to display the attributes for a single QA pair evaluation
    """
    
    print(f"""
    Question : {x['question']}

    Answer : {x['answer']}

    Context : {x['context']}

    Grade : {x['grade']}
    
    Description : {x['eval_description']}
    """)