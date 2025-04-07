# 🧙‍♂️ Ask the Wizarding World — A LangGraph Agent
Welcome to **Ask the Wizarding World**, a fun project built to learn and explore [LangGraph](https://langchain-ai.github.io/langgraph/) while diving into the magical universe of **Harry Potter**!

## 💡 Overview
This project creates an intelligent agent that answers questions about the Wizarding World. It uses vector search to retrieve passages from the Harry Potter books and combines that with a judgment step to decide whether to answer from the books or search the web.

## ✨ What It Does

This agent answers questions using a dynamic reasoning loop:

- 📚 Starts by retrieving info from a vector store built from the books using FAISS.

- ⚖️ Judges if the content is relevant.

- 🤖 If useful, it generates an answer with gemma-2b-it (via Groq).

- 🌐 If not, it does a web search, then:

    - ⚖️ Judges the new context

    - 🧠 Generates an answer

- ✅ Judges if the answer is helpful or should try again

- 🔁 Repeats until a helpful answer is found or decides the question is unsupported.

## 🪄 How to Use

```python
from IPython.display import display, Markdown
from retrievers.vector_retriever import VectorRetriever
from agents.agent import RunAgent

# Define paths to config file and source data
config_path = "path_to_config/config.yml"
path_to_data = "path_to_pdfs/data"

# Initialize the VectorRetriever and load data
vector_retriever = VectorRetriever(
    path_to_data=path_to_data, config_path=config_path
)
retriever = vector_retriever.load_data()

# Initialize the agent with the retriever and config file
agent_graph = RunAgent(
    retriever=retriever,
    config_path=config_path,
)

# Ask a question
question = "Who are the Dursleys?"
agent_output = agent_graph.run_agent(question=question)

print("\n")
display(Markdown(agent_output))
```

## 💻 Installation
1. Clone the repo:
    ```bash
    git clone https://github.com/thaismenegaldi/web_rag_agent.git
    cd web_rag_agent
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Add your `GROQ_API_KEY` and `SERPAPI_KEY` to your `.env` file.

4. Run your test as shown above.

## 🧰 Built With

- [LangGraph](https://langchain-ai.github.io/langgraph/)

- [LangChain](https://www.langchain.com/)

- `gemma-2b-it` via [Groq API](https://console.groq.com/docs/quickstart)

- [FAISS](https://github.com/facebookresearch/faiss)

- Python 🐍

## 📚 Dataset Source
The Harry Potter books used to build the vector store were obtained from the open repository:

🧹 [**Ginga1402/Harry-Potter-Dataset**](https://github.com/Ginga1402/Harry-Potter-Dataset)

All credit goes to the original author of that project. This dataset was used for educational purposes only.