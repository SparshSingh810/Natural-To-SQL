examples = [
    {
        "input": "List all customers in France with a credit limit over 20,000.",
        "query": "SELECT * FROM customers WHERE country = 'France' AND creditlimit > 20000;"
    },
    {
        "input": "Get the highest payment amount made by any customer.",
        "query": "SELECT MAX(amount) FROM payments;"
    },
    {
        "input": "Show product details for products in the 'Motorcycles' product line.",
        "query": "SELECT * FROM products WHERE productline = 'Motorcycles';"
    },
    {
        "input": "Retrieve the names of employees who report to employee number 1002.",
        "query": "SELECT firstname, lastname FROM employees WHERE reportsto = 1002;"
    },
    {
        "input": "List all products with a stock quantity less than 7000.",
        "query": "SELECT productname, quantityinstock FROM products WHERE quantityinstock < 7000;"
    },
    {
        'input': "what is price of '1968 Ford Mustang'",
        "query": "SELECT buyprice, msrp FROM products WHERE productname = '1968 Ford Mustang' LIMIT 1;"
    }
]

from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st


@st.cache_resource
def get_example_selector():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        embeddings,
        Chroma,
        k=2,
        input_keys=["input"],
    )
    return example_selector
