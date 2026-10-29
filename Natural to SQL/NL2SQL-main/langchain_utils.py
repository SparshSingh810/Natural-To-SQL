import os
import re
from dotenv import load_dotenv

load_dotenv()

db_user = os.getenv("db_user")
db_password = os.getenv("db_password")
db_host = os.getenv("db_host")
db_port = os.getenv("db_port", "5432")
db_name = os.getenv("db_name")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_classic.chains import create_sql_query_chain
from langchain_groq import ChatGroq
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.chat_history import InMemoryChatMessageHistory

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate

from table_details import table_chain as select_table
from prompts import final_prompt, answer_prompt

import streamlit as st


fix_query_prompt = PromptTemplate.from_template(
    "You wrote a PostgreSQL query that failed to execute. Fix it based on the error message.\n\n"
    "Original question: {question}\n"
    "Failed SQL query: {query}\n"
    "Database error: {error}\n\n"
    "Respond with ONLY the corrected raw SQL query and nothing else: no markdown code fences, "
    "no explanation, no leading label."
)


def clean_sql_query(query: str) -> str:
    query = query.strip()
    fence_match = re.match(r"^```(?:sql)?\s*\n?(.*?)\n?```$", query, re.DOTALL | re.IGNORECASE)
    if fence_match:
        query = fence_match.group(1).strip()
    else:
        query = re.sub(r"^```(?:sql)?\s*", "", query, flags=re.IGNORECASE)
        query = re.sub(r"```\s*$", "", query).strip()
    query = re.sub(r"^SQLQuery:\s*", "", query, flags=re.IGNORECASE).strip()
    return query


@st.cache_resource
def get_db():
    if not all([db_user, db_password, db_host, db_name]):
        raise RuntimeError(
            "Database credentials are missing. Set db_user, db_password, db_host, "
            "db_port and db_name via .env (local) or Streamlit secrets (deployed)."
        )
    db_uri = (
        f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        "?sslmode=require"
    )
    return SQLDatabase.from_uri(db_uri)


@st.cache_data
def get_schema_markdown():
    db = get_db()
    inspector = db._inspector
    lines = []
    for table in sorted(inspector.get_table_names()):
        pk_cols = set(inspector.get_pk_constraint(table).get("constrained_columns", []))
        fk_map = {}
        for fk in inspector.get_foreign_keys(table):
            for local_col, remote_col in zip(fk["constrained_columns"], fk["referred_columns"]):
                fk_map[local_col] = f"{fk['referred_table']}.{remote_col}"

        lines.append(f"**{table}**")
        for col in inspector.get_columns(table):
            name = col["name"]
            col_type = str(col["type"]).lower()
            marker = ""
            if name in pk_cols:
                marker += " 🔑"
            if name in fk_map:
                marker += f" → `{fk_map[name]}`"
            lines.append(f"- `{name}` ({col_type}){marker}")
        lines.append("")
    return "\n".join(lines)


@st.cache_resource
def get_chain():
    print("Creating chain")
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Set it via .env (local) or Streamlit secrets (deployed)."
        )

    db = get_db()
    llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY)
    generate_query = create_sql_query_chain(llm, db, final_prompt) | clean_sql_query
    execute_query = QuerySQLDataBaseTool(db=db)
    fix_query_chain = fix_query_prompt | llm | StrOutputParser() | clean_sql_query
    rephrase_answer = answer_prompt | llm | StrOutputParser()

    def execute_with_retry(inputs):
        query = inputs["query"]
        result = execute_query.invoke(query)
        if isinstance(result, str) and result.strip().lower().startswith("error"):
            fixed_query = fix_query_chain.invoke(
                {"question": inputs["question"], "query": query, "error": result}
            )
            fixed_result = execute_query.invoke(fixed_query)
            return {**inputs, "query": fixed_query, "result": fixed_result}
        return {**inputs, "result": result}

    chain = (
        RunnablePassthrough.assign(table_names_to_use=select_table)
        | RunnablePassthrough.assign(query=generate_query)
        | RunnableLambda(execute_with_retry)
        | rephrase_answer
    )

    return chain


def create_history(messages):
    history = InMemoryChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])
    return history


def invoke_chain(question, messages):
    chain = get_chain()
    history = create_history(messages)
    response = chain.invoke({"question": question, "top_k": 3, "messages": history.messages})
    history.add_user_message(question)
    history.add_ai_message(response)
    return response
