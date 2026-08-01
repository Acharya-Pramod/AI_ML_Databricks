# Enterprise Retrieval-Augmented Generation (RAG) Pipeline Using Databricks, LangChain, ChromaDB, Hugging Face Embeddings, and Groq LLM

## Introduction

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline using Databricks notebooks. The objective of the pipeline is to enable an LLM to answer questions accurately based on the contents of a specific PDF document rather than relying only on its pretrained knowledge. Instead of sending the entire PDF directly to the language model, the document is first processed into smaller semantic chunks, converted into numerical vector embeddings, stored in a vector database, and then searched efficiently using semantic similarity. Only the most relevant information retrieved from the document is provided to the LLM, which significantly improves response accuracy, reduces hallucinations, and enables the model to answer domain-specific questions.

---

# Overall Architecture

```
PDF
 │
 ▼
PyPDFLoader
 │
 ▼
Document Objects
 │
 ▼
RecursiveCharacterTextSplitter
 │
 ▼
Document Chunks
 │
 ▼
Sentence Transformer Embeddings
 │
 ▼
Chroma Vector Database
 │
 ▼
Semantic Similarity Search
 │
 ▼
Top-K Relevant Chunks
 │
 ▼
Groq Llama-3.3-70B
 │
 ▼
Generated Answer
```

---

# Step 1: Loading the PDF

The first stage of the pipeline involves loading the PDF document into memory. LangChain's **PyPDFLoader** is used because it automatically extracts textual information from each page while preserving metadata such as page numbers, source file path, and total page count. Instead of treating the PDF as a single long string, PyPDFLoader converts every page into an individual **Document object**, allowing page-level processing and traceability.

Each Document object contains two major components:

* **page_content** – the extracted text from the page.
* **metadata** – information such as page number, file name, and document source.

For example:

```
Document
|
|-- page_content
|
|-- metadata
      |-- page = 5
      |-- source = Reliance Q125 earnings transcript.pdf
```

This structured representation enables later stages of the pipeline to reference the original source of retrieved information.

---

# Step 2: Document Chunking

Large Language Models have limited context windows and cannot efficiently process entire documents that may contain hundreds of pages. Therefore, the document must be divided into smaller segments called **chunks**.

The project uses **RecursiveCharacterTextSplitter** from LangChain.

Parameters:

```
chunk_size = 1000

chunk_overlap = 200
```

The chunk size determines the maximum number of characters contained within each chunk.

Instead of splitting strictly every 1000 characters, the recursive splitter attempts to preserve semantic meaning by splitting in the following order:

1. Paragraphs

2. Sentences

3. New lines

4. Spaces

5. Characters

This hierarchical strategy minimizes the possibility of splitting important concepts across two chunks.

---

# Why Chunk Overlap?

A chunk overlap of 200 characters means that the last 200 characters of one chunk become the first 200 characters of the next chunk.

Example

```
Chunk 1

-------------------------

Company revenue increased...

Net profit increased...

EBITDA improved...

Expansion in retail...


Chunk 2

-------------------------

Expansion in retail...

Digital business...

AI investments...
```

Without overlap, important information located near chunk boundaries may become fragmented and lose contextual continuity.

Chunk overlap preserves contextual relationships across adjacent chunks and improves retrieval quality.

---

# Why 1000 Characters?

A chunk size of 1000 characters is commonly used because it provides an effective balance between semantic completeness and retrieval precision.

Small chunks

Pros

* higher retrieval precision

Cons

* may lose surrounding context

Large chunks

Pros

* richer context

Cons

* lower retrieval precision

1000 characters provides sufficient context while maintaining efficient semantic search.

---

# Step 3: Embedding Generation

After chunking, each chunk must be converted into numerical vectors.

Computers cannot compare natural language directly.

Instead, each chunk is transformed into a high-dimensional vector called an **embedding**.

The project uses

```
sentence-transformers/all-MiniLM-L6-v2
```

through

```
HuggingFaceEmbeddings
```

---

# Why This Embedding Model?

The selected model offers several advantages.

### Lightweight

Only approximately 22 million parameters.

---

### Fast

Embedding generation is extremely fast compared to larger embedding models.

---

### High Semantic Accuracy

It captures semantic similarity rather than simple keyword matching.

For example

```
"What is Jio's revenue?"
```

and

```
"How much money did Jio earn?"
```

produce very similar embeddings despite using different words.

---

### Open Source

No API cost.

Can run entirely locally.

Ideal for Databricks notebooks.

---

### Sentence Transformer Architecture

The model has been trained specifically for semantic search and information retrieval rather than text generation.

This makes it highly suitable for RAG systems.

---

# How Embeddings Work

Suppose one chunk contains

```
Reliance Retail recorded strong revenue growth...
```

The embedding model converts it into a vector similar to

```
[0.12,

-0.45,

0.98,

0.33,

...
384 dimensions]
```

Every chunk becomes one vector.

The same transformation is applied to user queries.

---

# Why Embeddings Instead of Keywords?

Traditional search

```
Revenue
```

matches

```
Revenue
```

only.

Embedding search understands meaning.

For example

```
Revenue

Sales

Income

Turnover
```

all produce nearby vectors.

This enables semantic search.

---

# Step 4: Creating the Vector Database

The project stores embeddings inside **ChromaDB**.

Chroma is an open-source vector database designed specifically for LLM applications.

Unlike SQL databases, Chroma indexes vectors according to semantic similarity rather than exact text.

Each stored record contains:

```
Chunk Text

↓

Embedding Vector

↓

Metadata
```

Example

```
Chunk

"Reliance Retail achieved..."

↓

Embedding

[0.12,-0.54,....]

↓

Metadata

Page 12

Document Name
```

---

# How Chroma Stores Data

Internally Chroma maintains:

* Original chunk text

* Embedding vectors

* Metadata

* Unique document IDs

The vectors are indexed for fast nearest-neighbor search.

The database is persisted locally:

```
persist_directory="/tmp/chroma_db"
```

or alternatively in a Databricks Volume for persistent storage.

---

# Step 5: Semantic Similarity Search

When the user asks

```
"What did management say about Jio?"
```

the query itself is embedded using the **same embedding model**.

The query vector is compared with every stored chunk vector.

The comparison uses **cosine similarity**, which measures the angle between vectors rather than their raw numerical values.

Cosine similarity ranges from:

* **1.0** → identical meaning
* **0** → unrelated
* **-1** → opposite direction (rare in embedding spaces)

The chunks with the highest similarity scores are selected.

This process is much more robust than keyword matching because it retrieves text based on semantic meaning rather than exact wording.

---

# Step 6: Retrieving the Most Relevant Chunks

Instead of passing the entire PDF to the language model, the retriever returns only the **top K** most relevant chunks.

For example:

```python
retriever = db.as_retriever(search_kwargs={"k": 5})
```

When the query is executed, the retriever returns the five chunks whose embeddings are closest to the query embedding. These chunks are concatenated to form the reference context supplied to the LLM.

---

# Step 7: Prompt Construction

The retrieved chunks are merged into a single context string and combined with the user's question.

The prompt instructs the LLM to answer **only from the supplied reference**, and to explicitly state when the answer cannot be found in the document. This grounding reduces hallucinations and keeps responses faithful to the PDF.

---

# Step 8: LLM Response Generation with Groq

The final prompt is sent to **Groq's Llama 3.3 70B Versatile** model through the OpenAI-compatible API.

The LLM does **not** search the PDF itself. It receives:

1. The user's question.
2. The retrieved document context.

Using these inputs, it generates a concise, natural-language answer based solely on the retrieved evidence.

---

# Advantages of the RAG Pipeline

Compared with sending the entire PDF directly to an LLM, this architecture offers several benefits:

* **Lower token usage** because only relevant chunks are sent.
* **Reduced hallucinations** by grounding responses in retrieved text.
* **Scalability** to thousands of documents.
* **Fast retrieval** using vector similarity instead of full-document scanning.
* **Traceability**, since retrieved chunks retain page metadata.
* **Flexibility**, allowing the same pipeline to support different LLM providers such as Groq, OpenAI, Anthropic, or Databricks Foundation Models.

---

# Conclusion

This project demonstrates a modern Retrieval-Augmented Generation workflow using Databricks, LangChain, Hugging Face sentence-transformer embeddings, ChromaDB, and Groq's Llama 3.3 model. The PDF is transformed into semantically meaningful chunks, each chunk is encoded into dense vector embeddings, and these vectors are stored in a vector database optimized for similarity search. At query time, the user's question is embedded using the same model, the most relevant chunks are retrieved using cosine similarity, and only this evidence is supplied to the LLM. This architecture enables accurate, context-aware question answering while reducing hallucinations and making efficient use of LLM context windows. It also provides a scalable foundation for extending the system to multiple documents, larger vector stores, or production-grade enterprise RAG applications.
