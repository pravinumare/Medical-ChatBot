from langchain.llms import CTransformers
from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_pinecone import PineconeVectorStore
from utils import download_embeddings
from dotenv import load_dotenv
import os
import streamlit as st
from streamlit_chat import message




embeddings = download_embeddings()

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Initializing the Pinecone
pc = Pinecone(api_key = PINECONE_API_KEY)

index_name = "medicalchatbot"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384, # Replace with your model dimensions
        metric="cosine", # Replace with your model metric
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) 
    )

# Loading the index
docsearch = PineconeVectorStore.from_existing_index(embedding=embeddings, index_name=index_name)


if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


system_prompt = (
    """      
    Use the given context to answer the question.
    If you don't know the answer, say you don't know.
    Use three sentence maximum and keep the answer concise.
    
    Context: {context}
    
    Do not exceed your answer more than 512 words.
    Only return the helpful answer below and nothing else.
    Helpful answer:"""
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)


# question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
# retriever=docsearch.as_retriever(search_kwargs={'k': 2})
# qa = create_retrieval_chain(retriever, question_answer_chain)


st.title("Langchain Chatbot")

response_container = st.container()
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Query: ", key="input")
    
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')


llm = CTransformers(model = r"C:\Users\Lenovo\Desktop\Projects\Generative AI\Medical ChatBot with new lang libs\model\llama-2-7b-chat.ggmlv3.q5_K_S.bin",
                    model_type = "llama",
                    config = {'max_new_tokens':512,
                              'temperature':0.8})

# conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt, llm=llm, verbose=True)

store = {}  # memory is maintained outside the chain

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversation = RunnableWithMessageHistory(llm, get_session_history)

if query:
    with st.spinner("typing..."):
        ...
        response = conversation.predict(input=f"Query:\n{query}")    # Context:\n {context} \n\n 
    st.session_state.requests.append(query)
    st.session_state.responses.append(response)

