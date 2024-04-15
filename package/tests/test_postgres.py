import os
import pytest
import unittest
import numpy as np

from promptflow.connections import CustomConnection
from pfazuredb.tools.postgrespg import vectorsearch
from dotenv import dotenv_values

# Get the absolute path to the root directory of your project
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load environment variables from .env file located at the root directory
config = dotenv_values(os.path.join(root_dir, ".env"))


@pytest.fixture
def test_postgres_connection() -> CustomConnection:
    test_postgres_connection = CustomConnection(
        name="postgres_connection",
        secrets={"my_key": "DUMMY"},  # unsupported
        configs={
            "conn_string": config["COSMOSDB_POSTGRES_CONN_STRING"],
        },
    )
    return test_postgres_connection


# @pytest.mark.skip(
#     reason="Need to provide a valid .env file for the Azure CosmosDB POSTGRES CONNECTION STRING AND TABLE_NAME"
# )
class TestTool:
    def test_vectorsearch(
        self,
        test_postgres_connection,
        table_name=config["TABLE_NAME"],
        search_type="vector",
        num_results=3,
        embeddings=[0.1, 0.2, 0.3] * 512,
        vectorsearch_method="L2",
        question = "None",
        filter_text = "None"
    ):
        result = vectorsearch(
            connection=test_postgres_connection,
            table_name=table_name,
            search_type=search_type,
            vectorsearch_method=vectorsearch_method,
            num_results=num_results,
            embeddings=embeddings,
            filter_text=filter_text,
            question = question
        )
        # Assert that the result is a list
        assert isinstance(result, list), "Result is not a list"
        # Assert that the result is not empty
        assert result, "Result is empty"
        # Assert that each element in the result list is a dictionary
        # assert all(isinstance(item, dict) for item in result), "Result contains non-dictionary elements"
        assert len(result) == num_results


# Run the unit tests
if __name__ == "__main__":
    unittest.main()
