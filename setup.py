from setuptools import setup, find_packages

setup(
    name='eyespy',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'dlib==19.24.1',
        'streamlit',
        # other dependencies
    ]
)