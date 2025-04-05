from prompts.prompt import Prompt

QEA_SYSTEM_PROMPT = Prompt(
    role="system",
    name="question_answering_system",
    prompt_template=(
        """
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        """
    )
)

QEA_USER_PROMPT = Prompt(
    role="user",
    name="question_answering_user",
    prompt_template=(
        """
        User's question: {question}

        Context: {context}
        """
    )
)
