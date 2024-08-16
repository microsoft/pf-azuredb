from promptflow import tool
from promptflow.connections import CustomConnection
from typing import Any, Dict, List, Optional, cast
import json
import warnings
from azure.cosmos import CosmosClient


@tool
def vectorsearch(
    connection: CustomConnection,
    db_name: str,
    container_name: str,
    num_results: int,
    embeddings: list,
    search_type: str,
    filter_text: str,
    search_index_name: str,
) -> str:

    # Initialize Cosmos DB client using the connection string from secrets
    COSMOS_NOSQL_CLIENT = CosmosClient.from_connection_string(
        connection.secrets["NOSQL_CONN_STRING"]
    )

    # Get the database client
    database = COSMOS_NOSQL_CLIENT.get_database_client(db_name)

    # Get the container client
    container = database.get_container_client(container_name)

    # Query to get a sample entry from the container
    output = container.query_items(
        query="SELECT * FROM c OFFSET 0 LIMIT 1", enable_cross_partition_query=True
    )
    sample_entry = list(output)[0]
    selected_keys = [
        k
        for k in sample_entry.keys()
        if k not in ["_rid", "_self", "_etag", "_attachments", "_ts", "@search.action"]
    ]

    non_vector_keys = [
        k for k in selected_keys if sample_entry[k] and not is_vector(sample_entry[k])
    ]
    non_vector_keys = ["c." + key for key in non_vector_keys]
    columns_str = ", ".join(non_vector_keys)

    # Perform vector search or filter_vector search based on search_type
    if search_type == "vector":
        output = container.query_items(
            query=f"SELECT TOP @num_results {columns_str}, VectorDistance(c.{search_index_name}, @embedding) AS SimilarityScore FROM c ORDER BY VectorDistance(c.{search_index_name},@embedding)",
            parameters=[
                {"name": "@embedding", "value": embeddings},
                {"name": "@num_results", "value": num_results * 1},
            ],
            enable_cross_partition_query=True,
        )
    elif search_type == "filter_vector":
        output = container.query_items(
            query=f"SELECT TOP @num_results {columns_str}, VectorDistance(c.{search_index_name}, @embedding) AS SimilarityScore FROM c WHERE ({filter_text}) ORDER BY VectorDistance(c.{search_index_name},@embedding)",
            parameters=[
                {"name": "@embedding", "value": embeddings},
                {"name": "@num_results", "value": num_results * 1},
            ],
            enable_cross_partition_query=True,
        )
    else:
        raise ValueError(
            "Invalid Input.Also note that Hybrid search not supported. Valid search_type: 'vector', 'filter_vector'"
        )

    ans = []
    while len(ans) < num_results and output:
        try:
            res = next(output)
            ans.append(res)
        except StopIteration:
            break
    
    # Sanity checking the results
    if not ans:
        warnings.warn("No results found for the given query")
    if len(ans) < num_results:
        warnings.warn(f"Only {len(ans)} results found for the given query")

    if "SimilarityScore" not in ans[0].keys():
        raise ValueError(
            "SimilarityScore not found in the output. Please check dimension match between the query embeddings and the embeddings in the container"
        )

    return ans


# Helper function to check if a value is a vector
def is_vector(value):
    return isinstance(value, list) and all(isinstance(i, (int, float)) for i in value)