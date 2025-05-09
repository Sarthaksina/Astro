from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="cosmic_market_oracle",
    version="0.1.0",
    description="AI-powered financial forecasting system integrating Vedic astrology with market data",
    author="Cosmic Market Oracle Team",
    author_email="info@cosmicmarketoracle.com",
    packages=find_packages(where="src"),
    package_dir={"":"src"},
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)