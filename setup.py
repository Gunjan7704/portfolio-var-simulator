# setup.py
from setuptools import setup, find_packages

setup(
    name="portfolio-var-simulator",
    version="1.0.0",
    description="Monte Carlo VaR Simulation for Futures Portfolio (Nifty, Gold, Crude Oil)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Gunjan Moradiya",
    url="https://github.com/gunjanmoradiya/portfolio-var-simulator",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "yfinance>=0.2.18",
        "plotly>=5.15.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "requests>=2.31.0",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)