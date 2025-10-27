from setuptools import setup, find_packages

setup(
    name="liquid_llm_vertex_pkg_stage1",
    version="0.1.0",
    description="Vertex AI training package for Liquid LLM Stage 1 distillation",
    author="Liquid LLM Team",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.40.0",
        "huggingface_hub>=0.23.0",
        "gcsfs>=2024.1.0",
        "tqdm>=4.66.0",
        "numpy>=1.26.0",
    ],
    python_requires=">=3.10",
)
