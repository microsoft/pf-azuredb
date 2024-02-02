from promptflow import tool
from promptflow.connections import CustomConnection


@tool
def vectorsearch(
    connection: CustomConnection,
    table_name: str,
    search_type: str,
    num_results: int,
    embeddings: list,
    filter_text: str,
) -> str:
    from pgvector.psycopg2 import register_vector
    from psycopg2 import Error
    from psycopg2 import sql
    import numpy as np
    import psycopg2

    connection = psycopg2.connect(connection.configs["conn_string"])
    register_vector(connection)
    if search_type == "vector":
        select_query = (
            f"SELECT id FROM {table_name} ORDER BY embedding <-> %s LIMIT {num_results}"
        )
    elif search_type == "hybrid":
        select_query = f"SELECT id FROM {table_name} where {filter_text} ORDER BY embedding <-> %s LIMIT {num_results}"
    else:
        raise Error(
            f"search_type{search_type} is not implemented. Please choose vector or hybrid"
        )

    cursor = connection.cursor()
    cursor.execute(select_query, (np.array(embeddings),))
    results = cursor.fetchall()
    top_ids = []
    for i in range(len(results)):
        top_ids.append(int(results[i][0]))

    connection.rollback()

    format_ids = ", ".join(["%s"] * len(top_ids))

    sql = f"SELECT CONCAT('productid: ', productid, ' ', 'score: ', score, ' ', 'text: ', text) AS concat FROM {table_name} WHERE id IN ({format_ids})"

    try:
        cursor.execute(sql, top_ids)
        top_rows = cursor.fetchall()
    except (Exception, Error) as e:
        print(f"Error executing SELECT statement: {e}")
    retrieved_results = [row[0] for row in top_rows]
    return retrieved_results