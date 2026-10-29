
from examples import get_example_selector
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate,PromptTemplate

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}\nSQLQuery:"),
        ("ai", "{query}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    example_selector=get_example_selector(),
    input_variables=["input","top_k"],
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a PostgreSQL expert. Given an input question, create a syntactically correct PostgreSQL query to run. Unless otherwise specified, quote identifiers only when necessary and never use MySQL-style backticks (`` ` ``) since PostgreSQL does not support them; use double quotes if quoting is required. It is fine, and often necessary, to use subqueries, JOINs, GROUP BY, HAVING, and aggregate functions for questions that require them (e.g. counting how many customers meet a condition based on a per-customer aggregate).\n\nRespond with ONLY the raw SQL query and nothing else: no markdown code fences (no ``` of any kind), no language tags, no explanation, and no leading label such as 'SQLQuery:'. The response must start directly with the SQL keyword (e.g. SELECT or WITH).\n\nWhen a question asks \"which <entity>\" (e.g. which employee, which customer, which product) based on a grouping, ranking, or join column that can contain NULLs, exclude rows where that column IS NULL before ranking/grouping, since NULL does not represent a real, named entity, unless the user is explicitly asking about missing or unassigned records.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries."),
        few_shot_prompt,
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}"),
    ]
)

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Note: a normal, successful SQL Result looks like a Python list of tuples, e.g. [(2,)] or [('Paris', 12)] — treat results in that form as valid data and answer normally from them.

Only treat the SQL Result as a failure if it is literally an error/exception message (e.g. it starts with "Error:", or names a database exception like OperationalError, ProgrammingError, SyntaxError, or contains a traceback). In that case, do NOT invent, guess, or hypothesize an answer — instead tell the user the query failed and briefly state what went wrong based on the error text.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)
