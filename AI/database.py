# import os 
# from langchain.text_splitter import CharacterTextSplitter
# # from langchain.embeddings.groq import groqEmbeddings
# from langchain.vectorstores import FAISS
# from api_key import api_key
# from groq import Groq

# with open("output.txt") as f:
#     state_of_the_union = f.read()
    
# text_splitter = CharacterTextSplitter(
#     separator="\n\n",
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len,
#     is_separator_regex=False
# )   

# texts=text_splitter.create_documents((state_of_the_union))
# print(texts[0])
# print(len(texts))

# #set OpenAI API key
# os.environ["GROQ_API_KEY"]= api_key
# db_faiss_path = os.path.join('db_faiss')
                             
# #directories creation
# os.makedirs(db_faiss_path, exist_ok=True)
# #verify directories
# os.path.exists(db_faiss_path)
# # embeddings = OpenAIEmbeddings()
# # vectorstore = FAISS.from_documents(texts,embedding=embeddings)
# # vectorstore.save_local(db_faiss_path)


# client = Groq()
# completion = client.chat.completions.create(
#     model="llama3-8b-8192",
#     messages=[texts],
#     temperature=1,
#     max_tokens=1024,
#     top_p=1,
#     stream=True,
#     stop=None,
# )

# for chunk in completion:
#     print(chunk.choices[0].delta.content or "", end="")
# import os
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.embeddings.openai import OpenAIEmbeddings
# from api_key import api_key
# from groq import Groq

# # Read file
# with open("output.txt") as f:
#     state_of_the_union = f.read()

# # Text splitting
# text_splitter = CharacterTextSplitter(
#     separator="\n\n",
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len,
#     is_separator_regex=False,
# )
# texts = text_splitter.create_documents([state_of_the_union])
# print(texts[0])
# print(len(texts))

# # Set API key
# os.environ["GROQ_API_KEY"] = api_key

# # FAISS vectorstore (optional)
# db_faiss_path = os.path.join('db_faiss')
# os.makedirs(db_faiss_path, exist_ok=True)

# embeddings = OpenAIEmbeddings(openai_api_key=api_key)
# vectorstore = FAISS.from_documents(texts, embeddings)
# vectorstore.save_local(db_faiss_path)

# # Groq chat completions
# client = Groq()
# messages = [{"role": "user", "content": text.page_content} for text in texts]

# completion = client.chat.completions.create(
#     model="llama3-8b-8192",
#     messages=messages,
#     temperature=1,
#     max_tokens=1024,
#     top_p=1,
#     stream=True,
#     stop=None,
# )

# # Print response
# for chunk in completion:
#     print(chunk.choices[0].delta.content or "", end="")
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformersEmbeddings
from langchain.vectorstores import FAISS
# from sentence_transformers import SentenceTransformer

# Load the pre-trained SentenceTransformer model (you might need to install it)
model_name = "all-mpnet-base-v2"  # Example model, choose a suitable one
# sentence_transformer = SentenceTransformer(model_name)

with open("output.txt") as f:
    state_of_the_union = f.read()

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False
)

texts = text_splitter.create_documents((state_of_the_union))
print(texts[0])
print(len(texts))

# Create embeddings using SentenceTransformers
# embeddings = SentenceTransformersEmbeddings(sentence_transformer)
# embeddings.fit(texts)  # This might take some time for the first run

# Create FAISS vector store
# vectorstore = FAISS.from_documents(texts, embedding=embeddings)

# Save the vector store locally (optional)
db_faiss_path = os.path.join("db_faiss")
os.makedirs(db_faiss_path, exist_ok=True)
# vectorstore.save_local(db_faiss_path)

# Currently, Llama 3 doesn't have a public API for text completion.
# You'll need to explore alternative libraries or services for completion tasks.

print("**Completion functionality not available yet**")