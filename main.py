# %%
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("API_KEY")

# %%
import pandas as pd

dataset = pd.read_excel('Sample_data.xlsx')

dataset

# %%
from langchain.docstore.document import Document
\

docs = []
for i, row in dataset.iterrows():
    content = " | ".join(f"{col}: {row[col]}" for col in dataset.columns)
    docs.append(Document(page_content=content))


from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)

documents = text_splitter.split_documents(docs)

# %%

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma

embeddings = OllamaEmbeddings(
    model = "llama3-groq-tool-use:8b"
)

db = Chroma.from_documents(docs, embeddings)


# %%
# query = ""
# docs_found = db.similarity_search(query)

# for d in docs_found:
#     print(d.page_content)

# %%
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
    Answer the follwing question based only on the provided context" 
    Think step by step before providing a detailed answer" 
    I will tip you 1000$ if the user finds the answer helpful."
    <context>
    {context} 
    </context> 
    Question: {input} 
    """
)




# %%
llm=ChatGroq(api_key=api_key, model=os.getenv("MODEL"))
from langchain.chains.combine_documents import create_stuff_documents_chain
document_chain = create_stuff_documents_chain(llm, prompt)

retriever = db.as_retriever(search_kwargs={"k": 100})


# %%
from langchain.chains import create_retrieval_chain

retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "what is the most reccuring incident and compare with barchart "})

# %%
response["answer"]

# %%
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

llm = ChatGroq(
        api_key=api_key,
        model = "llama-3.3-70b-versatile",
        temperature=0.7,
        max_retries=2,
        timeout=None
    )

prompt = ChatPromptTemplate.from_template('how are you')

chain = LLMChain(prompt=prompt, llm=llm)

result = chain.run({})

result
