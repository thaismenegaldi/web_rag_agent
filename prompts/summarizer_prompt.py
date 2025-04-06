from prompts.prompt import Prompt

SUMMARIZER_SYSTEM_PROMPT = Prompt(
    role="system",
    name="summarizer_system",
    prompt_template=(
        """
        You are an expert at analyzing JSON data and extracting the most
        relevant information to answer user questions.
        The JSON provided comes from a web search API.
        Your task is to carefully process this data and generate a
        user-friendly response based on the user’s query.

            - Identify the most relevant information in the JSON that addresses
             the user's question.
            - Rewrite the relevant information in a clear and natural way for
             human reading, ensuring readability and coherence.
            - Pay attention to the details in each JSON entry, do not let
             relevant data behind
        """
    ),
)

SUMMARIZER_USER_PROMPT = Prompt(
    role="user",
    name="summarizer_user",
    prompt_template=(
        """
        User's question:
        {question}

        Raw JSON data:
        {context}
        """
    ),
)
