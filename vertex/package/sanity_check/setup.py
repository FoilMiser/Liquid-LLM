from pathlib import Path
from setuptools import setup, find_packages

PACKAGE_ROOT = Path(__file__).parent
VERSION = (PACKAGE_ROOT / "sanity_check" / "VERSION").read_text().strip()

setup(
    name="vertex-sanity-check",
    version=VERSION,
    description="Vertex AI sanity and smoke tests for Stage-0 distilled model checkpoints.",
    author="Liquid LLM",
    packages=find_packages(),
    package_data={"sanity_check": ["VERSION"]},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3",
        "transformers>=4.44",
        "tokenizers>=0.19",
        "gcsfs>=2024.6.0",
        "huggingface_hub>=0.24",
        "tqdm>=4.66",
        "python-json-logger>=2.0.7",
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": ["sanity-check=sanity_check.cli:main"],
    },
)
