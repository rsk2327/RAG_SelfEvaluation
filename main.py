import pandas as pd
import numpy as np
import os
from io import StringIO 
import json
from enum import Enum
from collections import Counter

from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
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
from tqdm import tqdm 

from utils import *


def retrieve_context(docs, splitting_strategy, question, database, k = 4):
    """ Given a set of documents and a question, retrieves the most relevant 'chunks' of text from the docs that help answer the question

    Args:
        docs (List[str]): Represents the raw text extracted from each individual pdf
        splitting_strategy (List[Dict[str,str]] or Dict[str,str]): Specifies the splitting strategy to be used for splitting the docs. Can be a single strategy or a combination of strategies 
        question (str): Question for which the context needs to be retrieved
        database (str): Select vector database type. One of 'faiss','chroma','pinecone'
        k (int, optional): Represents the number of chunks to be returned. Defaults to 4.

    Returns:
        List[str]: List of chunks extracted from docs that are most suitable for answering the question

    """

    if isinstance(splitting_strategy, dict):
        ## Only 1 splitting strategy provided
        splits = retrieve_splits(docs, splitting_strategy)
    else:
        ## Multiple splitting strategies provided
        splits = []

        for strategy_ in splitting_strategy:
            splits += retrieve_splits(docs, strategy_)

    # Instantiates a langchain retriever for the given database type, populated with the computed text splits/chunks
    retriever = get_retriever(splits, database,k)

    # Extracts the top chunks/splits corresponding to the question by calling the retriever
    context = retriever.invoke(question)

    return context



def retrieve_splits(docs, strategy):
    """Helper function to retrieve_context that retrieves the splits/chunks corresponding to a single splitting strategy

    Args:
        docs (List[str]): Represents the raw text extracted from each individual pdf
        strategy (Dict[str,str]): Specifies the splitting strategy to be used for splitting the docs. 


    Returns:
        List[str]: List of chunks extracted from docs that are most suitable for answering the question
    """

    if strategy['type'] == 'recursive':
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=strategy['params']['chunk_size'], 
                                                       chunk_overlap=strategy['params']['chunk_size']
                                                      )
        
    elif strategy['type'] == 'semantic':
        text_splitter = SemanticChunker(OpenAIEmbeddings())
    else:
        raise ValueError("Incorrect splitter type provided. Select one of 'recursive', 'semantic' ")
        
    splits = text_splitter.split_documents(docs)
    return splits




class Experiment(object):

    def __init__(self, df, docs, splitting_strategy, database, k, llm_eval_retries = 3, num_evals = 1):
        """Initiates an experiment object to run LLM based self evaluation of the context text retrieved from vector database

        Args:
            df (pd.DataFrame): Dataframe containing the original evaluation questions
            docs (List[str]): Represents the raw text extracted from each individual pdf
            splitting_strategy (List[Dict[str,str]] or Dict[str,str]): Specifies the splitting strategy to be used for splitting the docs. Can be a single strategy or a combination of strategies 
            database Select vector database type. One of 'faiss','chroma','pinecone'
            k (_type_): Represents the number of chunks to be returned. Defaults to 4.
            llm_eval_retries (int, optional): Represents the number of times to run LLM self eval pipeline incase of errors. Errors are typically thrown when LLM output doesnt match specified JSON format. Defaults to 3.
            num_evals (int, optional): Represents the number of times to run the LLM eval pipeline for a given question-answer pair
        """

        self.df = df
        self.docs = docs
        self.splitting_strategy = splitting_strategy
        self.database = database
        self.k = k
        self.llm_eval_retries = llm_eval_retries
        self.num_evals = num_evals

        self.qa_prompt = get_qa_prompt()
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        self.llm_eval_chain = get_llm_eval_chain()
        

    def run_test(self):

        if isinstance(self.splitting_strategy, dict):
            ## Only 1 splitting strategy provided
            splits = retrieve_splits(self.docs, self.splitting_strategy)
        else:
            ## Multiple splitting strategies provided
            splits = []
    
            for strategy_ in self.splitting_strategy:
                splits += retrieve_splits(self.docs, strategy_)
        
        retriever = get_retriever(splits, self.database,self.k)

        rag_chain = self.qa_prompt | self.llm

        output_df = []
    
        for question in tqdm(self.df['relevant questions']):
    
            context = format_docs(retriever.invoke(question))
            rag_output = rag_chain.invoke({'question':question, 'context':context})
            answer = rag_output.content

            eval_output = self.get_self_eval_results(question, answer)
            
            output_df.append([question, answer, context, eval_output['grade'], eval_output['description']])
    
        output_df = pd.DataFrame(output_df, columns=['question','answer','context','grade','eval_description'])

        self.save_results(output_df, './results/')

        print(f"Accuracy : {self.get_accuracy(output_df)}" )


    def get_accuracy(self, df):

        df['final_grade'] = df.grade.apply(lambda x : Counter(x).most_common()[0][0])

        accuracy = np.round(np.sum(df.final_grade=='Correct')/len(df),2)

        return accuracy





    def get_self_eval_results(self, question, answer):

        eval_output = {'grade':[], 'description':[]}

        for i in range(self.num_evals):
            retries = 0
            while retries < self.llm_eval_retries:
                try:
                    output = self.llm_eval_chain.invoke({'question':question, 'answer':answer})
                    eval_output['grade'].append(output['grade'])
                    eval_output['description'].append(output['description'])
                    break
                except:
                    retries += 1
                    if retries == self.llm_eval_retries:
                        raise ValueError('LLM Output contains wrong format')

        return eval_output






    
    def save_results(self, df, folder):

        if isinstance(self.splitting_strategy, dict):
           
            if self.splitting_strategy['type']=='recursive':
                chunk_size = self.splitting_strategy['params']['chunk_size']
                chunk_overlap = self.splitting_strategy['params']['chunk_overlap']
                df.to_csv(os.path.join(folder, f'output_{chunk_size}_{chunk_overlap}_{self.k}_{self.database}.csv'),index_label=False, index=False)
            else:
                df.to_csv(os.path.join(folder, f'output_1_1_{self.k}_{self.database}.csv'),index_label=False, index=False)
                
        else:
            df.to_csv(os.path.join(folder, f'output_multi_{self.k}_{self.database}.csv'),index_label=False, index=False)

        
        


        
            
    


    

    

    