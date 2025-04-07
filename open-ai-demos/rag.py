# rag_basic.py

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

import os
from dotenv import load_dotenv

load_dotenv()

OPEN_AI_KEY = os.getenv('OPEN_AI_KEY')
os.environ["OPENAI_API_KEY"] = OPEN_AI_KEY


# 2. Sample text (this could be loaded from a file)
text = """
LangChain is a framework for developing applications powered by language models.
It enables retrieval-augmented generation (RAG), which combines LLMs with external knowledge.
RAG improves accuracy by pulling in relevant facts from a document store before generating responses.
"""

# 3. Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
documents = text_splitter.create_documents([text])

# 4. Embed and store in a vector database (Chroma)
embedding = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embedding)

# 5. Create a retriever
retriever = vectorstore.as_retriever()

# 6. Build the RAG chain
llm = ChatOpenAI(model="gpt-3.5-turbo")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 7. Ask a question
query = "What is LangChain used for?"
result = qa_chain(query)

# 8. Print the answer
print("Answer:", result['result'])

# (Optional) View source docs
print("\nSources:")
for doc in result['source_documents']:
    print("-", doc.page_content.strip())
