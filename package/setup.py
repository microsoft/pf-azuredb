from setuptools import find_packages, setup

PACKAGE_NAME = "pf-azuredb"

setup(
    name=PACKAGE_NAME,
    author="Hossein K. Heris, applied ai team@azure data",
    version="0.0.1",
    description="Package for use in promptflow for vector search in azure db",
    packages=find_packages(),
    entry_points={
        "package_tools": ["azuredb = pfazuredb.tools.utils:list_package_tools"],
    },
    include_package_data=True,  # This line tells setuptools to include files from MANIFEST.in
)
