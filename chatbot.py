from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()
import os
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ["Gemini"]=os.getenv("GEMINI_API_KEY")

doc=PyPDFLoader("ai.pdf")
document=doc.load()

text=RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=400)
model_data=text.split_documents(document)
data=model_data[0].page_content

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings


embedding=GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.environ["Gemini"]
)

text_embed=embedding.embed_query(data)
# text_embed

from langchain.vectorstores import FAISS

vectorstore=FAISS.from_documents(model_data,embedding)
query_search=vectorstore.similarity_search("{input}")
retriever=vectorstore.as_retriever(search_kwargs={"k":3})
# query_search[0].page_content

from langchain_google_genai import ChatGoogleGenerativeAI

llm=ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    temperature=0.7,
    google_api_key=os.environ["Gemini"]
)

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate



prompt_tempelate=""" 
You are an expert at giving answers to the questions so give a accurate and short answer for the questions asked by the user.If it has two questions continue the next question on seperate line :

{context}


Questions:

 """

prompt=ChatPromptTemplate.from_messages([
    ("system",prompt_tempelate),
    ("user","{input}"),
])

ques_chain=create_stuff_documents_chain(llm,prompt)
rag_chain=create_retrieval_chain(retriever,ques_chain)

question="What are ad solar energy"





## Streamlit 

st.title("Curious about AI? Let’s chat!")

input_text=str(st.text_input("Ask Question?"))


if input_text!="":
    responce=rag_chain.invoke({"input":input_text})
    st.subheader("Answer:")
    st.success(responce["answer"]) 
    # st.write(responce['answer'])
    # st.markdown(f"```markdown\n{responce['answer']}\n```")
    # st.markdown('Answer:{responce["answer"]}', height=300)
