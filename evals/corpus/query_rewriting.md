Query rewriting cleans up the user question before retrieval. Typical steps strip conversational filler such as "can you please" and expand abbreviations so the retriever sees the same vocabulary the documents use.

Acronym expansion helps when documents spell out full terms. A question about LLM latency can match a passage that only says large language model, and a question about the FAQ can match frequently asked questions.

Rewriting can also hurt. Expanding an identifier prefix or a product code into generic words may dilute the exact token that made keyword search precise, so every rewrite rule should be checked against a labeled eval set.
