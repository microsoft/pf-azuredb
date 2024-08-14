from setuptools import find_packages, setup

PACKAGE_NAME = "pf-azuredb"

setup(
    name=PACKAGE_NAME,
    author="Hossein K. Heris and applied ai team@azure data",
    version="1.1.0",
    description="Package for use in promptflow for vector search in azure db",
    packages=find_packages(),
    entry_points={
        "package_tools": ["azuredb = pfazuredb.tools.utils:list_package_tools"],
    },
    include_package_data=True,  # This line tells setuptools to include files from MANIFEST.in
    install_requires=[
        "numpy==1.24.4",
        "psycopg2-binary==2.9.6",
        "pymongo==4.6.1",
        "pgvector==0.2.0",
        "azure-cosmos==4.7.0",
    ],
)
