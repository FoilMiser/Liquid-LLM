from setuptools import find_packages, setup

setup(
    name="liquid_llm_vertex_pkg_stage1",
    version="0.1.0",
    description="Liquid LLM Stage-1 Vertex AI training package",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.40.0",
        "huggingface_hub>=0.23.0",
        "torch==2.4.0",
        "numpy",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": ["liquid-stage1=stage1.cli:main"],
    },
)
