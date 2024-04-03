from promptflow import tool
from promptflow.connections import CustomConnection
from typing import Any, Dict, List, Optional, cast
import json
import warnings


@tool
def vectorsearch(
    connection: CustomConnection,
    db_name: str,
    collection_name: str,
    num_results: int,
    embeddings: list,
    filter_query: str = "{}",
    search_type: str = "vector",
    embedding_key: str = "contentVector",
) -> str:
    from pymongo import MongoClient

    uri = connection.configs["AZURE_COSMOSDB_MONGODB_URI"]
    mongo_client = MongoClient(uri)

    db = mongo_client[db_name]
    collection = db[collection_name]

    if search_type == "vector" and filter_query:
        warnings.warn(
            "\nwarning:\nfilter_query is being ignored for search_type=vector:\n"
        )
    elif search_type in ("filter_vector", "hybrid") and not filter_query:
        warnings.warn(
            ":\nwarning:\nfilter_vector/hybrid is being selected but no filter is provided. In this case, only vector search applies:\n"
        )
        pass

    if search_type == "vector":
        params = {"vector": embeddings, "path": embedding_key, "k": num_results}
    elif search_type == "filter_vector" or search_type == "hybrid":
        if search_type == "hybrid":
            warnings.warn(
                "hybrid for search type input is assumed to be 'filter_vector'", Warning
            )
        filters_query_loaded = json.loads(filter_query)

        params = {
            "vector": embeddings,
            "path": embedding_key,
            "k": num_results,
            "filter": filters_query_loaded,
        }
    else:
        raise ValueError("Invalid Input. Valid search_type: 'vector', 'filter_vector'")

    query_field = {"$search": {"cosmosSearch": params, "returnStoredSource": True}}

    pipeline = [
        query_field,
        {
            "$project": {
                "similarityScore": {"$meta": "searchScore"},
                "document": "$$ROOT",
            }
        },
    ]

    results = list(collection.aggregate(pipeline))
    undesired_keys = {embedding_key, "_id"}
    retrieved_results = [
        {k: v for k, v in res["document"].items() if k not in undesired_keys}
        for res in results
    ]
    return retrieved_results
