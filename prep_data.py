from langchain.document_loaders import ReadTheDocsLoader
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

file_path = '/Users/aa/Desktop/RAG TEST 1/.venv/best_practices.txt'


with open(file_path, 'r', encoding='utf-8') as file:
    docs = file.read()




tokenizer = tiktoken.get_encoding('cl100k_base')

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

token_count = tiktoken_len(docs)
print(token_count)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,  # number of tokens overlap between chunks
    length_function=tiktoken_len,
    separators=['\n\n', '\n', ' ', '']
)

chunks = text_splitter.split_text(docs)



formatted_chunks = []
for i, chunk in enumerate(chunks):
    chunk_data = {
        "id": i,
        "text": chunk,
        "source": "best_practices.txt"
    }
    formatted_chunks.append(chunk_data)





with open('train.jsonl', 'w') as f:
    for chunk in formatted_chunks:
        f.write(json.dumps(chunk) + '\n')


formatted_chunks = []

with open('train.jsonl', 'r') as f:
    for chunk in f:
        formatted_chunks.append(json.loads(chunk))




