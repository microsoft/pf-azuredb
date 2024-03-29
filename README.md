# Project

Source repository for pf-azuredb library hosted on pypi. 

This library is a promptflow tool/pluggin to do vector search from azure cosmosdb mongodb vCore and azure cosmosdb postgres. 

A sample usage of this library can be found at the following repo: https://github.com/microsoft/promptflow-rag-project-template

## Install library from this source

Prerequisites:
```
pip install promptflow
pip install pytest pytest-mock
```
change directory to the package folder. 
```
cd .\package 
```
The run the following command in the package root directory to build your tool:

```
python setup.py sdist bdist_wheel
```

This will generate a tool package "pf-azuredb-<version>.tar.gz" and associated whl file inside the "dist" folder. 

To install the package in your python environment, please use the following command. Note to replace version with the actual version. 

```
pip install .\dist\pf_azuredb-<version>-py3-none-any.whl --force-reinstall
```

At this stage, a new version is installed in your python environment and you can switch to your promptflow project. By clicking + on the promptflow, you will see these tools are added and ready to use. 

## Maintainer

As the maintainer of this project, please make a few updates:

- Improving this README.MD file to provide a great experience
- Updating SUPPORT.MD with content about this project's support experience
- Understanding the security reporting process in SECURITY.MD
- Remove this section from the README

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
