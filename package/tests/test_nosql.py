import os
import pytest
import unittest
import numpy as np

from promptflow.connections import CustomConnection
from pfazuredb.tools.nosql import vectorsearch
from dotenv import dotenv_values

# Get the absolute path to the root directory of your project
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load environment variables from .env file located at the root directory
config = dotenv_values(os.path.join(root_dir, ".env"))


@pytest.fixture
def test_nosql_connection() -> CustomConnection:
    test_nosql_connection = CustomConnection(
        name="nosql_connection",
        secrets={"NOSQL_CONN_STRING": config["COSMOS_DB_NOSQL_CONN_STRING"]},
    )
    return test_nosql_connection


# @pytest.mark.skip(
#     reason="Need to provide a valid .env file for the Azure NoSQL CONNECTION STRING, DB_NAME, and CONTAINER_NAME"
# )
class TestTool:
    def test_vectorsearch(
        self,
        test_nosql_connection,
        db_name=config["DB_NAME_NOSQL"],
        container_name=config["CONTAINER_NAME_NOSQL"],
        num_results=3,
        embeddings=[0.1, 0.2, 0.3] * 512,
        search_type="vector",
        filter_text="None",
        search_index_name="contentVector",
    ):
        result = vectorsearch(
            connection=test_nosql_connection,
            db_name=db_name,
            container_name=container_name,
            num_results=num_results,
            embeddings=embeddings,
            search_type=search_type,
            filter_text=filter_text,
            search_index_name=search_index_name,
        )
        # Assert that the result is a list
        assert isinstance(result, list), "Result is not a list"
        # Assert that the result is not empty
        assert result, "Result is empty"
        # Assert that each element in the result list is a dictionary
        # assert all(isinstance(item, dict) for item in result), "Result contains non-dictionary elements"
        assert len(result) == num_results
        # Assert that the result contains the similarity key.
        # note: this case resolve cases where embedding dimension mismatch
        assert "SimilarityScore" in result[0].keys()
        # Make sure similarity score is calculated
        assert result[0]["SimilarityScore"] is not None


# Run the unit tests
if __name__ == "__main__":
    unittest.main()
