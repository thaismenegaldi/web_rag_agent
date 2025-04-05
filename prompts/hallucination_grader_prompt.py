from prompts.prompt import Prompt

HALLUCINATION_SYSTEM_PROMPT = Prompt(
    role="system",
    name="hallucination_system",
    prompt_template=(
        """
        You are a grader determining whether an answer is grounded in a
        given set of facts.
        Assess if the answer is fully supported by the provided facts and
        assign a binary score:
            - "yes" if the answer is entirely supported.
            - "no" if the answer contains inaccuracies, unsupported claims, or
            missing justification.
        Provide the binary score as a JSON with a single key 'score' and no
        additional explanation.
        """
    )
)

HALLUCINATION_USER_PROMPT = Prompt(
    role="user",
    name="hallucination_user",
    prompt_template=(
        """
        Here are the facts:
        {documents}

        Here is the answer:
        {generation}
        """
    )
)
