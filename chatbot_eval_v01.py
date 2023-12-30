# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 11:24:25 2023

@author: Sarvenaz Chaeibakhsh
"""

import numpy as np
import langchain
import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, DocArrayInMemorySearch

from langchain.chat_models import ChatOpenAI, AzureChatOpenAI, PromptLayerChatOpenAI, QianfanChatEndpoint


def load_htmls():
    ''' load and reads all html docs in one pass'''
    all_files = [f for f in os.listdir('data') if f.endswith('.html')] # every html file in the folder
    docs = []
    for file in all_files:
        doc = UnstructuredHTMLLoader('data/'+file).load() # loading each document
        parsed_doc_name = file.split('/')[0].split('_') 
        first_name = parsed_doc_name[0]
        last_name = parsed_doc_name[1]
        doc_type = parsed_doc_name[2].split('.')[0] 
        # print(f'name:{first_name}, family_name:{last_name}, doc_type = {doc_type}')
        # adding to each document metadata for later easier search
        doc[0].metadata['name'] = ' '.join([first_name, last_name])
        doc[0].metadata['doc_type'] = doc_type # -> may not use it, keeping it for now
        docs.extend(doc)
    return docs

def get_names(docs = None):
    ''' get a list of all names'''
    if docs is None:
        docs = load_htmls()
    return list(set([doc.metadata['name'] for doc in docs]))


def name_based_retriever(docs = None, name = 'All', embeddings = OpenAIEmbeddings(), chunk_size = 1000, chunk_overlap = 10):
    '''
    Creats custom retriever based on the name chosen by the user
    '''
    if docs == None:
        docs = load_htmls()
    
    if name == 'All':
        print('all')
        documents = docs
    else:
        print(f'name is {name}')
        documents = [doc for doc in docs if doc.metadata['name'] == name] 
    # split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index = True)
    texts = text_splitter.split_documents(documents)

    #vector database
    db = Chroma.from_documents(texts, embeddings)

    # expose this index in a retriever interface
    client_filter = {'client_name': {'$eq': name}}
    retriever = db.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )
    
    return retriever, texts

if __name__ == '__main__':
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature = 0.0) # language model
    # llm = QianfanChatEndpoint(temperature = 0.0)
    
    embeddings = OpenAIEmbeddings()
    
    # r, t = name_based_retriever(docs = None, name = 'Mariann Avocado') # test
    
    name = 'Robert King'
    name = 'Jerry Smith'
    r, t = name_based_retriever(docs = None, name = name)
    qa = RetrievalQA.from_chain_type(
        llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature = 0),
        chain_type="stuff",
        retriever=r,
        return_source_documents=False,
        verbose=False,
    )
    
    print(qa('What industry does Velvet Throat work at?')['result'])
    q = f'where does {name} work at?'
    print(q, qa(q)['result'])
    
    #%%
    # Initializing feedback function
    from trulens_eval.feedback.provider import OpenAI as fOpenAI
    import numpy as np
    
    # Imports main tools:
    from trulens_eval import TruChain, Feedback, Huggingface, Tru
    from trulens_eval.schema import FeedbackResult
    tru = Tru()
    tru.reset_database()
    
    # Initialize provider class
    provider = fOpenAI()
    
    # select context to be used in feedback. the location of context is app specific.
    from trulens_eval import TruLlama
    
    context_selection = TruLlama.select_source_nodes().node.text
    
    from trulens_eval.feedback import Groundedness
    grounded = Groundedness(groundedness_provider=provider)
    # Define a groundedness feedback function
    f_groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons,
                  name="Groundedness"
                )
        .on(context_selection)
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
        )
    #     .aggregate(grounded.grounded_statements_aggregator)
    # )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     