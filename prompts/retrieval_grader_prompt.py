from prompts.prompt import Prompt

RETRIEVAL_SYSTEM_PROMPT = Prompt(
    role="system",
    name="retrieval_system",
    prompt_template=(
        """
        You are a grader assessing relevance of a retrieved document to a
        user question.
        If the document contains any information or keywords related to the
        user question, grade it as relevant.
        This is a very lenient test - the document does not need to fully
        answer the question to be considered relevant.

        Give a binary score 'yes' or 'no' to indicate whether the document is
        relevant to the question.
        Also provide a brief explanation for your decision.

        Return your response as a JSON with two keys:
        'score' (either 'yes' or 'no') and 'explanation'.
        """
    )
)

RETRIEVAL_USER_PROMPT = Prompt(
    role="user",
    name="retrieval_user",
    prompt_template=(
        """
        Here is the retrieved document:
        {document}

        Here is the user question:
        {question}
        """
    )
)
