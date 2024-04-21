import os
from langchain_openai import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from pinecone import Pinecone
import datasets
from datasets import load_dataset
from langchain.embeddings.openai import OpenAIEmbeddings    
from tqdm.auto import tqdm 


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or "key here"

test = os.environ["OPENAI_API_KEY"]

chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-4'
)



messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hi AI, how are you today?"),
    AIMessage(content="I'm great thank you. How can I help you?"),
    HumanMessage(content="I'd like to have some financial assistance.")
]





###############################################################
######NOW BUILD RAG PULLING DATASET PREVIOULSY PREPARED AND UPLOADED TO HUGGING FACE

file_path = 'train.jsonl'

#dataset = load_dataset('json', data_files=file_path, split='train')
dataset = load_dataset( "JAAR90/best_practices", split="train")



api_key = os.getenv("PINECONE_API_KEY") or "key here"

# configure client
pc = Pinecone(api_key=api_key)

from pinecone import ServerlessSpec

spec = ServerlessSpec(
    cloud="aws", region="us-west-2"
)



import time

index_name = 'best-practices'
existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]

if index_name not in existing_indexes:
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=1536, 
        metric='dotproduct',
        spec=spec
    )
    
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)


index = pc.Index(index_name)
time.sleep(1)
# view index stats
#print(index.describe_index_stats())



embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")



###here we start the embedding process


data = dataset.to_pandas()  # this makes it easier to iterate over the dataset

batch_size = 100

for i in tqdm(range(0, len(data), batch_size)):
    i_end = min(len(data), i+batch_size)
    # get batch of data
    batch = data.iloc[i:i_end]
    # generate unique ids for each chunk
    ids = [f"{x['id']}" for i, x in batch.iterrows()]
    # get text to embed
    texts = [x['text'] for _, x in batch.iterrows()]
    # embed text
    embeds = embed_model.embed_documents(texts)
    # get metadata to store in Pinecone
    metadata = [
        {'text': x['text'],
         'source': x['source'],
         'id': x['id']} for i, x in batch.iterrows()
    ]
    # add to Pinecone
    index.upsert(vectors=zip(ids, embeds, metadata))

#print(index.describe_index_stats())    
    



from langchain.vectorstores import Pinecone
text_field = "text"  # the metadata field that contains our text

# initialize the vector store object
vectorstore = Pinecone(
    index, embed_model.embed_query, text_field
)    


query = " what would you recommend to Consider  before borrowing from my retirement plan? maybe using a 401k credit card"



def augment_prompt(query: str):
    # get top 3 results from knowledge base
    results = vectorstore.similarity_search(query, k=3)
    # get the text from the results
    source_knowledge = "\n".join([x.page_content for x in results])
    # feed into an augmented prompt
    augmented_prompt = f"""Using the contexts below, answer the query.

    Contexts:
    {source_knowledge}

    Query: {query}"""
    return augmented_prompt


# create a new user prompt
prompt = HumanMessage(
    content=augment_prompt(query)
)
# add to messages
messages.append(prompt)

res = chat(messages)


print(res.content)
prompt = HumanMessage(
    content="what would you recommend to Consider  before borrowing from my retirement plan? maybe using a 401k credit card"
)

res = chat(messages + [prompt])


print("answer of chatgpt 4")
print("##################################################################################")
print(res.content)

print("##################################################################################")