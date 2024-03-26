from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
import os
import time

# Initialize OpenAI and Qdrant clients with API keys globally
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

def ensure_collection_exists(collection_name):
    # Check if the collection already exists, and create it if not
    existing_collections = [collection.name for collection in qdrant_client.get_collections().collections]
    if collection_name not in existing_collections:
        qdrant_client.create_collection(
            collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

def embed_text_and_upsert(collection_name, chunk_folder):
    # Directory where text chunks are stored
    files = os.listdir(chunk_folder)

    # Iterate over each file in the chunk folder
    for file_name in files:
        file_path = os.path.join(chunk_folder, file_name)
        with open(file_path, 'r') as file:
            text = file.read()

        # Create embedding for the text and upsert into Qdrant
        embedding = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        ).data[0].embedding
        time.sleep(1)

        # Upload to qdrant vectorDB
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=int(file_name),  # Assuming file names are just the integer IDs
                    payload={"text": text},
                    vector=embedding
                ),
            ]
        )

collection_name = "sc4052-lecture"
chunk_folder = 'chunk/'

ensure_collection_exists(collection_name)
embed_text_and_upsert(collection_name, chunk_folder)
