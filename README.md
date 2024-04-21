

<h1>RAG-AI-Implementation-Financw</h1>



<h2>Description</h2>
This is a project that implements Retrieval Augmented Generation techniques on top of Open AI's API, that allows storing NL type data in a Vector Database to the use as context for specific question to the LLM.
It takes an article on best fiancial practices, split it in chunks to then assign meta tabs to each chunk, than then are stored in an Index in Pinecone(Vector Database).
This vector data is ennbeded and retrieved based on their semmantic meanings (using Open AI ada 002 model) to give specific context to the LLM based on the users question.
This allos the LLM to have memory and provide a personalized assitance
<br />


<h2>Languages and Utilities Used</h2>

- <b>Python</b> 

<h2>Environments and Tech Used </h2>

- <b>Pinecone Database</b> 
- <b>Hugging face</b> 
- <b>Open AI API</b>
- <b>Open AI gpt-4 MOODEL</b>
- <b>Open AI text-embedding-ada-002 MOODEL</b>



<!--
 ```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```
--!>
