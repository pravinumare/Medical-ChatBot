from langchain.llms import CTransformers
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_pinecone import PineconeVectorStore
from utils import download_embeddings
from dotenv import load_dotenv
import os
import streamlit as st



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



llm = CTransformers(model = r"C:\Users\Lenovo\Desktop\Projects\Generative AI\Medical ChatBot with new lang libs\model\llama-2-7b-chat.ggmlv3.q5_K_S.bin",
                    model_type = "llama",
                    config = {'max_new_tokens':512,
                              'temperature':0.8})

question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
retriever=docsearch.as_retriever(search_kwargs={'k': 2})
qa = create_retrieval_chain(retriever, question_answer_chain)


st.set_page_config(page_title="Medical Chatbot",
                   page_icon="🏥",
                   layout="centered")

st.header("Medical Chatbot 🩺")

query = st.text_input("Enter your question")

button = st.button("Get Answer")


if button:
    with st.spinner("Loading please wait..."):

        response = qa.invoke({"input" : query})

        print(response['answer'])
        
        st.text_area(label="Response", value=response['answer'])
