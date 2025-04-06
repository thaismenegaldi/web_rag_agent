import logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="\n%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def log_agent_step(message: str) -> None:
    """
    Log a message indicating the step of the agent's process.

    Parameters
    ----------
    message : str
        The message to be logged.
    """
    logging.info(f"[AGENT STEP] {message}")
