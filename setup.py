from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="quest5",
    version="0.0.1",
    author="Tharindu Ranasinghe",
    author_email="rhtdranasinghe@gmail.com",
    description="Translation Quality Estimation with T5 ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TharinduDR/QuesT5",
    packages=find_packages(exclude=("examples", "docs", )),
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "requests",
        "tqdm>=4.47.0",
        "regex",
        "transformers>=4.2.0",
        "datasets",
        "scipy",
        "scikit-learn",
        "seqeval",
        "tensorboardx",
        "pandas",
        "tokenizers",
        "wandb",
        "streamlit",
        "sentencepiece",
    ],
)
