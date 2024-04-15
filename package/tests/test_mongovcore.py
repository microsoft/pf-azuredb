import os
import pytest
import unittest
import numpy as np

from promptflow.connections import CustomConnection
from pfazuredb.tools.mongodbvcore import vectorsearch
from dotenv import dotenv_values

# Get the absolute path to the root directory of your project
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load environment variables from .env file located at the root directory
config = dotenv_values(os.path.join(root_dir, ".env"))


@pytest.fixture
def test_mongo_connection() -> CustomConnection:
    test_mongo_connection = CustomConnection(
        name="mongovcore_connection",
        secrets={"my_key": "DUMMY"},  # unsupported
        configs={
            "AZURE_COSMOSDB_MONGODB_URI": config["COSMOS_DB_MONGO_URI"],
        },
    )
    return test_mongo_connection


# @pytest.mark.skip(
#     reason="Need to provide a valid .env file for the Azure CosmosDB MongoDB URI, DB_NAME, and COLLECTION_NAME"
# )
class TestTool:
    def test_vectorsearch(
        self,
        test_mongo_connection,
        db_name=config["DB_NAME"],
        collection_name=config["COLLECTION_NAME"],
        num_results=3,
        embeddings=[0.1, 0.2, 0.3] * 512,
    ):
        result = vectorsearch(
            connection=test_mongo_connection,
            db_name=db_name,
            collection_name=collection_name,
            num_results=num_results,
            embeddings=embeddings,
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
