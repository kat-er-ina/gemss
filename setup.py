from setuptools import setup, find_packages

setup(
    name="gemss",
    version="0.1.0",
    description="GEMSS: Gaussian Ensemble for Multiple Sparse Solutions.",
    author="Katerina Henclova",
    author_email="katerina.henclova@datamole.ai",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.23.0",
        "pandas>=2.0.0,<3.0.0",
        "torch>=2.0.0,<3.0.0",
        "plotly>=5.15.0",
        "scikit-learn>=1.3.0,<2.0.0",
        "jupyter>=1.0.0,<2.0.0",
        "pyarrow>=22.0.0",
        "tqdm>=4.65.0",
        "ipywidgets>=8.0.0",
    ],
    python_requires=">=3.11",
)
