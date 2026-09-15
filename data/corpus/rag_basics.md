# Retrieval-Augmented Generation (RAG)

RAG combines information retrieval with text generation. A corpus of documents is
embedded and indexed. At query time, the system retrieves the most relevant
chunks and passes them to a language model as context so answers stay grounded
in source material.

Typical pipeline:
1. Ingest and chunk documents
2. Embed chunks into vectors
3. Store vectors in a searchable index
4. Embed the user question
5. Retrieve top-k similar chunks
6. Generate an answer conditioned on those chunks
