from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    embedding = model.encode(text)
    return embedding.tolist()

if __name__ == "__main__":
    test_text = "What is the leave policy of OrgMind?"
    vector = get_embedding(test_text)
    print(f"Text: {test_text}")
    print(f"Vector dimensions: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")