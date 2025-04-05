from prompt import Prompt

SEARCH_SYSTEM_PROMPT = Prompt(
    role="system",
    name="search_system",
    prompt_template=(
        """
        You are an expert at finding relevant information to answer
        user question. Criticize the context based on user's question, and
        find the most usefull field from the context, a list of keys, that
        can have the user's answer. If you don't identify any field usefull,
        returns 'organic_results'.

        Return a JSON with a single key 'field' with the chosen ones and
        no preamble or explanation.

        Examples:
        Question: CONMEBOL Libertadores future games?
        Answer: {{"field": "sports_results"}}

        Question: Brazilian's 2025 holidays?
        Answer: {{"field": "answer_box"}}
        """
    )
)

SEARCH_USER_PROMPT = Prompt(
    role="user",
    name="search_user",
    prompt_template=(
        """
        User's question:
        {question}

        Context:
        {context}
        """
    )
)
