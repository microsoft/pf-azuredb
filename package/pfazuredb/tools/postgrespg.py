from promptflow import tool
from promptflow.connections import CustomConnection

@tool
def vectorsearch(
    connection: CustomConnection,
    table_name: str,
    search_type: str,
    num_results: int,
    embeddings: list,
    vectorsearch_method: str,
    filter_text: str,
) -> list:
    from pgvector.psycopg2 import register_vector
    from psycopg2 import Error
    import numpy as np
    import psycopg2
    import json

    pgconnection = psycopg2.connect(connection.configs["conn_string"])
    register_vector(pgconnection)

    if vectorsearch_method == "L2":
        distance_operator = "<->"
    elif vectorsearch_method == "Cosine":
        distance_operator = "<=>"
    elif vectorsearch_method == "Inner":
        distance_operator = "<#>"
    else:
        raise Error(
            f"Vector search method '{vectorsearch_method}' is not implemented. Please choose one of: L2, Cosine, Inner."
        )
    
    if search_type == "vector":
        select_query = (
            f"SELECT * FROM {table_name} ORDER BY embedding {distance_operator} %s LIMIT {num_results}"
        )
    elif search_type == "hybrid":
        select_query = f"SELECT * FROM {table_name} where {filter_text} ORDER BY embedding {distance_operator} %s LIMIT {num_results}"
    else:
        raise Error(
            f"search_type{search_type} is not implemented. Please choose vector or hybrid"
        )

    cursor = pgconnection.cursor()
    cursor.execute(select_query, (np.array(embeddings),))
    results = cursor.fetchall()

    retrieved_results = []
    for row in results:
        row_data = {}
        for idx, col in enumerate(cursor.description):
            # Check if the value is serializable, otherwise convert it to a string
            if isinstance(row[idx], np.ndarray):
                row_data[col.name] = row[idx].tolist()
            else:
                row_data[col.name] = row[idx]
        retrieved_results.append(row_data)

    cursor.close()
    pgconnection.close()

    return retrieved_results