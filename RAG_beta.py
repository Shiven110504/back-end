#!/usr/bin/env python3

from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma  
from langchain.chains import RetrievalQA

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
documents = loader.load()

embeddings = OpenAIEmbeddings()

vector_store = Chroma.from_documents(documents, embedding=embeddings) 

retriever = vector_store.as_retriever()

prompt_template = PromptTemplate(template="Based on the retrieved information, answer the question: {topic}", input_variables=["topic"])

llm = ChatOpenAI(model_name="gpt-3.5-turbo")
retrieval_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, prompt=prompt_template)

user_query = input("Please enter your query: ")
retrieved_info = retrieval_chain.run({"topic": user_query})

response = f"User Query: {user_query}\nRetrieved Information: {retrieved_info}"
print(response)