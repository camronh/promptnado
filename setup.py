from setuptools import setup, find_packages

setup(
    name="promptnado",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "langsmith",
        "pydantic>=1.8,<3.0",
        "python-dotenv",
        "langchain",
        "langchain-openai",
        "langgraph>=0.1.15",
    ],
    python_requires=">=3.9.0,<3.12",
)