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
    
    # # select context to be used in feedback. the location of context is app specific.
    # from trulens_eval import TruLlama
    
    # context_selection = TruLlama.select_source_nodes().node.text
    
    # from trulens_eval.feedback import Groundedness
    # grounded = Groundedness(groundedness_provider=provider)
    # # Define a groundedness feedback function
    # f_groundedness = (
    #     Feedback(grounded.groundedness_measure_with_cot_reasons,
    #               name="Groundedness"
    #             )
    #     .on(context_selection)
    #     .on_output()
    #     .aggregate(grounded.grounded_statements_aggregator)
    #     )

    qa_relevance = Feedback(provider.relevance).on_input_output()
    f_context_relevance = (
        Feedback(provider.relevance_with_cot_reasons,
                 name = 'Q&A Relevance')
        .on_input()
        .on_output()
    # .aggregate(np.mean)
    )
    tru_recorder = TruChain(qa,
                            app_id='Chain1_ChatApplication',
                            feedbacks=[
                                # f_qa_relevance,
                                f_context_relevance,
                                # f_groundedness
                                ])
    
    # %%
    with tru_recorder as recording:
        llm_response = qa(q)['result']

    print(llm_response)
    
    # The record of the ap invocation can be retrieved from the `recording`:
    rec = recording.get() # use .get if only one record
    
    records, feedback = tru.get_records_and_feedback(app_ids=["Chain1_ChatApplication"])
    # %%
    from langchain_core.runnables import RunnablePassthrough
    from langchain import hub
    from langchain_core.output_parsers.string import StrOutputParser
    from langchain.prompts import ChatPromptTemplate

    
    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI()
    
    retrieval_chain = (
        {"context": r, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    # print(retrieval_chain.invoke(f"where did {name} work?"))
    
    
    # %%
from langchain.chains import LLMChain
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, PromptTemplate

full_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        template=
        "Provide a helpful response with relevant background information for the following: {prompt}",
        input_variables=["prompt"],
    )
)

chat_prompt_template = ChatPromptTemplate.from_messages([full_prompt])


chain = LLMChain(llm=llm, prompt=chat_prompt_template, verbose=True)

print(chain(q))
    
    
# %%
'''----------------------------------------------------------------
                         RAGAS 
---------------------------------------------------------------'''
from ragas.metrics import faithfulness, answer_relevancy, context_relevancy
from ragas.langchain import RagasEvaluatorChain # langchain chain wrapper to convert a ragas metric into a langchain

eval_metrics_list = [faithfulness, answer_relevancy, context_relevancy]
# make eval chains
eval_chains = {
    m.name: RagasEvaluatorChain(metric=m) 
    for m in eval_metrics_list
}   
    

for name, eval_chain in eval_chains.items():
    score_name = f"{name}_score"
    print(f"{score_name}: {eval_chain({'query':q, 'source_documents':t, 'result':llm_response})[score_name]}")
    
# %%
# Visualizing the metrics for a set of questions

import matplotlib.pyplot as plt
def plot_metrics_with_values(metrics_dict, title='RAG Metrics'):
    """
    Plots a bar chart for metrics contained in a dictionary and annotates the values on the bars.
    Args:
    metrics_dict (dict): A dictionary with metric names as keys and values as metric scores.
    title (str): The title of the plot.
    """
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    plt.figure(figsize=(10, 6))
    bars = plt.barh(names, values, color='skyblue')
    # Adding the values on top of the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01,  # x-position
                 bar.get_y() + bar.get_height() / 2,  # y-position
                 f'{width:.4f}',  # value
                 va='center')
    plt.xlabel('Score')
    plt.title(title)
    plt.xlim(0, 1)  # Setting the x-axis limit to be from 0 to 1
    plt.show()
    
plot_metrics_with_values({'faithfulness_score': 1.0,
                          'answer_relevancy_score': 0.9468557283238227,
                          'context_relevancy_score': 0.025}, "Base Retriever ragas Metrics")

# %%
# Generating a set of sample questions and answers to test score aggregation
eval_questions = [
    'Where does Jerry work?',
    'What industry or industries is Jerry involved in?',
    'Is Jerry a charitable person?',
    'Who are Jerrys family memebers?',
    ]
eval_answers = [qa(q)['result'] for q in eval_questions]
# %%
metric_tracker = {}
for metric, eval_chain in eval_chains.items():
    metric_name = f'{metric}_score'
    metric_tracker[metric_name] = []
    for ind, q in enumerate(eval_questions):
        metric_tracker[metric_name].append(eval_chain({'query':q, 'source_documents':t, 'result':eval_answers[ind]})[metric_name])
    metric_tracker[f'{metric_name}_avg'] = np.mean(metric_tracker[metric_name]) # Calculating mean over each metric for an overall assesment

# %%
# Plottingt the avg values

metric_avg = {k:metric_tracker[k] for k in [metric.name + '_score_avg' for metric in eval_metrics_list]}
plot_metrics_with_values(metric_avg, title = 'Avg RAG Metrics')