from setuptools import setup, find_packages

setup(
    name="eyespy",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit>=1.24.0",
        "setuptools>=61.0",
        "wheel>=0.40.0",
    ],
    python_requires=">=3.8",
)