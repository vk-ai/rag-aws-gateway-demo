Hit rate at k asks one question per query: did at least one relevant chunk appear in the top k results? It is easy to explain but ignores how many relevant chunks were found and where they ranked.

Recall at k is the fraction of all relevant chunks for a query that appear in the top k results. When a question needs two supporting passages, recall shows whether the retriever found both of them.

Mean reciprocal rank averages 1 / rank of the first relevant chunk across queries. A relevant chunk at position one scores 1.0, position two scores 0.5, and a miss scores zero.

Normalized discounted cumulative gain, nDCG, uses graded relevance labels. Highly relevant chunks near the top earn more gain than partially relevant ones, and the total is divided by the ideal ordering so scores fall between zero and one.
