import os
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import streamlit as st
from operator import itemgetter
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from typing import List

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY)


@st.cache_data
def get_table_details():
    # Read the CSV file into a DataFrame
    table_description = pd.read_csv("database_table_descriptions.csv")

    # Iterate over the DataFrame rows to build the table descriptions block
    table_details = ""
    for index, row in table_description.iterrows():
        table_details = table_details + "Table Name:" + row['Table'] + "\n" + "Table Description:" + row['Description'] + "\n\n"

    return table_details


class Table(BaseModel):
    """Table in SQL database."""

    name: str = Field(description="Name of table in SQL database.")


class Tables(BaseModel):
    """All SQL tables that might be relevant to the user question."""

    tables: List[Table] = Field(description="List of relevant tables in the SQL database.")


def get_tables(tables: Tables) -> List[str]:
    return [table.name for table in tables.tables]


table_details = get_table_details()
table_details_prompt = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
The tables are:

{table_details}

Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""

table_selection_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", table_details_prompt),
        ("human", "{question}"),
    ]
)

table_chain = (
    {"question": itemgetter("question")}
    | table_selection_prompt
    | llm.with_structured_output(Tables)
    | get_tables
)
