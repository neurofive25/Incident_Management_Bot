# %%
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")
model = os.getenv("MODEL")

# %%
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "translate the {text} into {language}"
)

fmt = prompt.format(language = "tamil", text="Hello Welcome")

fmt

# %%
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.chains import LLMChain

llm = ChatGroq(
    api_key=api_key,
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_retries=2,
    timeout=None,
)


chat_history = InMemoryChatMessageHistory()

memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=chat_history,
    return_messages=True
)

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{query}")
])

conversation_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

response = conversation_chain.run({"query": "What is AI?"})

response2 = conversation_chain.run({"query": "what question i ask before"})
print(response2)


# %%
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

df = pd.read_excel("../data/Sample_data.xlsx")

docs = []
for i, row in df.iterrows():
    content = " | ".join(f"{row[col]}" for col in df.columns)
    docs.append(Document(page_content = content))

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)

documents = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

documents

# %%
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores.faiss import FAISS

# Build FAISS index
db = FAISS.from_documents(documents, embeddings)

# Create retriever
retriever = db.as_retriever(search_kwargs = {'k': len(documents)+1})

# Prompt template must include {context} and {input}
prompt = PromptTemplate(
    input_variables=["context", "input"],
    template="""
You are an assistant. Use the provided context to answer the question.

If the user asks for a chart/plot/diagram:
- Return a JSON object with these fields:
  - "x_axis": label for the X-axis
  - "y_axis": label for the Y-axis
  - "x_data": give values for X-axis
  - "y_data": give value for Y-axis
  - "values_type": description of what values to retrieve
  - "chart_type": suggested chart type ("bar", "line", "scatter", "pie")


- note:
  - give json data as a string in a way we can directly load using json.load()
  - ascending and decesending order not comes under chart unless if explicitly mention
  - give only as json data for plotting if a keyword pie chart, scatter chart, 
  bar chart is mentioned in the question
  
Rules:
- Only output valid JSON.
- If the user does not explicitly ask for a chart, return .


Context:
{context}

Question:
{input}
"""
)

# Stuff retrieved documents into the prompt
document_chain = create_stuff_documents_chain(llm, prompt)

# Build retrieval chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Run query
response = retrieval_chain.invoke({"input": "give data related to singapore hub"})

print(response["answer"])


# %%
import json
data = json.loads(response['answer'])

x = data['x_data']
x

# %%
import matplotlib.pyplot as plt

def plot_from_result(data):

    chart_type = data["chart_type"]

    x = data["x_data"]
    y = data["y_data"]

    plt.figure(figsize=(8, 5))

    if chart_type == "bar":
        plt.bar(x, y)
    elif chart_type == "line":
        plt.plot(x, y, marker="o")
    elif chart_type == "scatter":
        plt.scatter(x, y)
    elif chart_type == "pie":
        plt.pie(y, labels=x, autopct="%1.1f%%")
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    if chart_type != "pie":
        plt.xlabel(data["x_axis"])
        plt.ylabel(data["y_axis"])
        plt.title(f"{data["x_axis"]} vs {data["y_axis"]}")
        plt.grid(True)
    else:
        plt.title(f"{data["y_axis"]} vs {data["x_axis"]}")

    plt.show()

plot_from_result(data)

MODEL = "llama-3.3-70b-versatile"
