from setuptools import setup, find_packages

setup(
    name="knowledge_padi",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "Flask==2.3.2",
        "PyPDF2==3.0.1",
        "langchain==0.0.275",
        "astra-sdk==0.3.0",
        "langchain-openai==0.1.3",
        "python-dotenv==1.0.0",
        "pytest==7.4.2",
        "requests==2.32.3",
    ],
    description="Knowledge Padi AI - A Retrieval-Augmented Generation chatbot",
    developer="Oluwakayode Nicholas",
    developer_email="oluwakayodenicholas1@gmail.com",
    url="https://github.com/Nico3783/knowledge-padi.git",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
