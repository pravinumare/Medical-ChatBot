from flask import Flask, render_template, request
from langchain.llms import CTransformers
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_pinecone import PineconeVectorStore
from utils import download_embeddings
from dotenv import load_dotenv
import os


app = Flask(__name__)


embeddings = download_embeddings()

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Initializing the Pinecone
pc = Pinecone(api_key = PINECONE_API_KEY)

index_name = "medicalchatbot"

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
                              'temperature':0.2})

question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
retriever=docsearch.as_retriever(search_kwargs={'k': 2})
qa = create_retrieval_chain(retriever, question_answer_chain)


@app.route('/')
def index():
    return render_template('chat.html')


@app.route('/get', methods=['GET', 'POST'])
def chat():
    user_input = request.form['msg']
    print(user_input)
    result = qa.invoke({'input':user_input})
    print("Response : ", result['answer'])
    return str(result['answer'][9:])


if __name__ == "__main__":
    app.run(debug=True)