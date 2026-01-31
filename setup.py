
from setuptools import setup, find_packages

setup(
    name="query_classifier",
    version="0.1.0",
    description="A generic intent classification library using Semantic Routing and LLMs",
    author="Antigravity",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "sentence-transformers",
        "transformers",
        "torch",
        "ollama",
        "uagents"
    ],
    python_requires=">=3.8",
)
