from setuptools import find_packages, setup

setup(
    name="xai-credit-scoring",
    version="0.2.0",
    description="Explainable, fairness-audited credit scoring (synthetic data)",
    author="Prakhar Prakarsh",
    author_email="prakharprakarsh@gmail.com",
    packages=find_packages(),
    python_requires=">=3.11",
)
