from prompt import Prompt

ROUTER_SYSTEM_PROMPT = Prompt(
    role="system",
    name="router_system",
    prompt_template=(
        """
        You are an expert at routing user questions to either a vector_store or
        web search based on their topic.
            - Use "vector_store" for questions related to  Harry Potter, the
            wizarding world, magical creatures, spells, characters,
            or similar themes.
            - Use "web_search" for all other topics.

        Output your decision as a JSON object with a single key "datasource"
        and no additional explanation.

        Examples:
        Question: Who is Hermione Granger?
        Answer: {{"datasource": "web_search"}}

        Question: What is the Accio spell used for??
        Answer: {{"datasource": "vector_store"}}

        Question: What are the La Liga next fixtures?
        Answer: {{"datasource": "vector_store"}}

        Question: What is prompt engineering?
        Answer: {{"datasource": "vector_store"}}
        """
    )
)

ROUTER_USER_PROMPT = Prompt(
    role="user",
    name="router_user",
    prompt_template=(
        """
        Question to route:
        {question}
        """
    )
)
