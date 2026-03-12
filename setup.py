from setuptools import setup, find_packages

setup(
    name="xai-credit-scoring",
    version="1.0.0",
    description="Explainable & Fair Credit Scoring — EU AI Act Compliant",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "shap>=0.43.0",
        "lime>=0.2.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.5.0",
        "streamlit>=1.28.0",
        "aif360>=0.6.0",
        "fairlearn>=0.9.0",
        "structlog>=23.2.0",
        "pyyaml>=6.0",
        "joblib>=1.3.0",
    ],
)
