from promptflow import tool
from promptflow.connections import CustomConnection


@tool
def vectorsearch(
    connection: CustomConnection,
    table_name: str,
    search_type: str,
    num_results: int,
    question: str,
    embeddings: list,
    vectorsearch_method: str,
    filter_text: str,
) -> list:
    from pgvector.psycopg2 import register_vector
    from psycopg2 import Error
    import numpy as np
    import psycopg2
    import json

    # establish connection
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
        if vectorsearch_method == "L2":
            select_query = f"SELECT *, (embedding {distance_operator} %s) AS score FROM {table_name} ORDER BY score LIMIT {num_results}"
        elif vectorsearch_method == "Cosine":
            select_query = f"SELECT *, 1 - (embedding {distance_operator} %s) AS score FROM {table_name} ORDER BY score DESC LIMIT {num_results}"
        elif vectorsearch_method == "Inner":
            select_query = f"SELECT *, (embedding {distance_operator} %s) * -1 AS score FROM {table_name} ORDER BY score DESC LIMIT {num_results}"
    elif search_type == "filter_vector":
        if vectorsearch_method == "L2":
            select_query = f"SELECT *, (embedding {distance_operator} %s) AS score FROM {table_name} where {filter_text} ORDER BY score LIMIT {num_results}"
        elif vectorsearch_method == "Cosine":
            select_query = f"SELECT *, 1 - (embedding {distance_operator} %s) AS score FROM {table_name} where {filter_text}  ORDER BY score DESC LIMIT {num_results}"
        elif vectorsearch_method == "Inner":
            select_query = f"SELECT *, (embedding {distance_operator} %s) * -1 AS score FROM {table_name} where {filter_text} ORDER BY score DESC LIMIT {num_results}"
    elif search_type == "hybrid":
        # Adapted from RRF sample: https://github.com/pgvector/pgvector-python/blob/master/examples/hybrid_search_rrf.py
        select_query = f"""
        WITH semantic_search AS (
            SELECT *, RANK () OVER (ORDER BY embedding {distance_operator} %(embedding)s) AS rank
            FROM {table_name}
            ORDER BY embedding {distance_operator} %(embedding)s
            LIMIT 20
        ),
        keyword_search AS (
            SELECT *, RANK () OVER (ORDER BY ts_rank_cd(to_tsvector('english', content), question) DESC)
            FROM {table_name}, plainto_tsquery('english', %(question)s) question
            WHERE to_tsvector('english', content) @@ question
            ORDER BY ts_rank_cd(to_tsvector('english', content), question) DESC
            LIMIT 20
        )
        SELECT
            COALESCE(semantic_search.id, keyword_search.id) AS id,
            COALESCE(1.0 / (%(k)s + semantic_search.rank), 0.0) +
            COALESCE(1.0 / (%(k)s + keyword_search.rank), 0.0) AS score,
            semantic_search.*
        FROM semantic_search
        FULL OUTER JOIN keyword_search ON semantic_search.id = keyword_search.id
        ORDER BY score DESC
        LIMIT {num_results}
        """
    else:
        raise Error(
            f"search_type{search_type} is not implemented. Please choose vector, filter_vector, or hybrid"
        )

    cursor = pgconnection.cursor()

    if search_type != "hybrid":
        cursor.execute(select_query, (np.array(embeddings), ))
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
            del row_data['embedding']
            retrieved_results.append(row_data)
    else:
        cursor.execute(select_query, {'question': question, 'embedding': (np.array(embeddings), ), 'k': 60})
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
            del row_data['embedding']
            row_data['score'] = float(row_data['score'])
            retrieved_results.append(row_data)

    cursor.close()
    pgconnection.close()

    return retrieved_results