from langchain.llms import CTransformers
# from langchain_community.llms import CTransformers
from utils import download_embeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_pinecone import PineconeVectorStore
import os
import streamlit as st
from dotenv import load_dotenv

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

def get_response(user_query, chat_history):

    # template = """
    # You are a helpful assistant. Answer the following questions considering the history of the conversation:

    # Chat history: {chat_history}

    # User question: {user_question}
    # """

    # prompt = ChatPromptTemplate.from_template(template)

    system_prompt = (
    """You are a helpful assistant. Answer the following questions considering the history of the conversation:
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

    llm = CTransformers(model = r"C:\Users\Lenovo\Desktop\Projects\Generative AI\Medical ChatBot with new lang libs\model\llama-2-7b-chat.ggmlv3.q5_K_S.bin",
                    model_type = "llama",
                    config = {'max_new_tokens':512,
                              'temperature':0.7})
        
    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retriever=docsearch.as_retriever(search_kwargs={'k': 2})
    qa = create_retrieval_chain(retriever, question_answer_chain)    
    
    # return qa.invoke({
    #     "chat_history": chat_history,
    #     "user_question": user_query,
    # })
    return qa.invoke({'input':user_query})


# app config
st.set_page_config(page_title="Streamlit Chatbot", page_icon="🤖")

st.header("ChatBot")


# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

    
# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.chat_history)
        st.write(response['answer'])
        # st.markdown()
    st.session_state.chat_history.append(AIMessage(content=response))

