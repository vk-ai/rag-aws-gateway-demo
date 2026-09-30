Reciprocal rank fusion merges several ranked lists into one. Each document earns 1 / (k + rank) from every list it appears in, and the sums decide the final order. The constant k, often 60, dampens the influence of the very top positions.

Okapi BM25 is a keyword ranking function. It rewards documents that contain rare query terms, saturates repeated term frequency with the k1 parameter, and normalizes for document length with the b parameter.

Dense retrieval embeds the query and every chunk into vectors and ranks chunks by cosine similarity. It tolerates word order changes but can miss exact identifiers, part numbers, and acronyms that appear in only one chunk.

Hybrid search runs a keyword channel and a vector channel side by side and fuses their rankings, so a chunk that is strong in either channel can reach the top of the combined list.
