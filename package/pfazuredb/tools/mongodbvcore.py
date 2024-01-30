from promptflow import tool
from promptflow.connections import CustomConnection


@tool
def vectorsearch(
    connection: CustomConnection,
    db_name: str,
    collection_name: str,
    num_results: int,
    embeddings: list,
) -> str:
    from pymongo import MongoClient

    uri = connection.configs["AZURE_COSMOSDB_MONGODB_URI"]
    mongo_client = MongoClient(uri)
    query_embedding = embeddings

    db = mongo_client[db_name]
    collection = db[collection_name]
    pipeline = [
        {
            "$search": {
                "cosmosSearch": {
                    "vector": query_embedding,
                    "path": "contentVector",  # embedding_key,
                    "k": num_results,  # , "efsearch": 40 # optional for HNSW only
                },
                "returnStoredSource": True,
            }
        },
        {
            "$project": {
                "similarityScore": {"$meta": "searchScore"},
                "document": "$$ROOT",
            }
        },
    ]
    results = list(collection.aggregate(pipeline))

    return [res["document"]["content"] for res in results]
