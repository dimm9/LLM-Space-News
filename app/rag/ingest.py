import os
import pandas as pd
import pickle
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
OVERLAP = 120
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "spacenews.csv")
INDEX_PATH = os.path.join(DATA_DIR, "spacenews.index")
META_PATH = os.path.join(DATA_DIR, "spacenews.pkl")

def simple_chunk(text, chunk_chars=280, overlap=40):
    out = []
    i = 0
    if not isinstance(text, str): return []
    while i < len(text):
        j = min(chunk_chars+i, len(text))
        out.append(text[i:j])
        if j == len(text): break
        i = max(0, j-overlap)
    return out

def embed_texts(texts, embedder, batch_size=64):
    return embedder.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True).astype("float32")


def ingest():
    # read csv
    if not os.path.exists(CSV_PATH):
        print(f"No file {CSV_PATH}")
        return

    df_csv = pd.read_csv(CSV_PATH)
    df_csv = df_csv.head(500)
    df_csv = df_csv.dropna(subset=["content", "title"]).reset_index(drop=True)
    print(f"Loaded {len(df_csv)} articles from the CSV file. (truncated)")

    # build chunks
    rows = []
    for _, row in df_csv.iterrows():
        text = row['content']
        for k, txt_chunk in enumerate(simple_chunk(text, CHUNK_SIZE, OVERLAP)):
            if txt_chunk.strip():
                rows.append({
                    "source": row['url'],
                    "title": row['title'],
                    "date": row['date'],
                    "page": 1,
                    "chunk_id": k,
                    "chunk": txt_chunk
                })

    chunks_df = pd.DataFrame(rows)
    print(f"Generated {len(chunks_df)} chunks.")

    # embeddings and FAISS
    print(f"Loading model {MODEL_NAME}...")
    embedder = SentenceTransformer(MODEL_NAME)

    chunks_list = chunks_df["chunk"].tolist()
    print("Creating embeddings...")
    embs = embed_texts(chunks_list, embedder, batch_size=64)

    print("Building texy with FAISS...")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    # saving for retriever
    faiss.write_index(index, INDEX_PATH)

    chunks_data = chunks_df.to_dict('records')
    with open(META_PATH, "wb") as f:
        pickle.dump(chunks_data, f)

    print(f"Index and data saved in the folder {DATA_DIR}")


if __name__ == "__main__":
    ingest()