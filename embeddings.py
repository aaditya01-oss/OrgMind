from sentence_transformers import SentenceTransformer

# Multilingual model (same 384 dimensions as before, so the database column does not change).
# It was trained with prefixes: "passage: " for stored text and "query: " for questions.
MODEL_NAME = "intfloat/multilingual-e5-small"
model = SentenceTransformer(MODEL_NAME)


def get_embedding(text):
    """Embedding for text that will be STORED (document chunks)."""
    return model.encode("passage: " + text, normalize_embeddings=True).tolist()


def get_query_embedding(text):
    """Embedding for a user's QUESTION."""
    return model.encode("query: " + text, normalize_embeddings=True).tolist()


def get_embeddings(texts, batch_size=16):
    """Many passages at once (faster; used when re-embedding)."""
    vectors = model.encode(["passage: " + t for t in texts], batch_size=batch_size,
                           normalize_embeddings=True)
    return [v.tolist() for v in vectors]


if __name__ == "__main__":
    vector = get_query_embedding("What is the leave policy of OrgMind?")
    print(f"Model: {MODEL_NAME} | dimensions: {len(vector)} | max tokens: {model.max_seq_length}")