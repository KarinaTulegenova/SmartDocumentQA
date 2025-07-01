# python3
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

loader = PyPDFLoader("data/example.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = Chroma.from_documents(texts, embedding)

retriever = db.as_retriever()
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

while True:
    query = input("Задайте вопрос: ")
    if query.lower() in ["выход", "exit", "quit"]:
        break
    answer = qa.run(query)
    print(f"Ответ: {answer}")
