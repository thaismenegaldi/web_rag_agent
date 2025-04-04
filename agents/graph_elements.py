from typing_extensions import TypedDict
from typing import List
import logging
from langchain_core.vectorstores.base import VectorStoreRetriever
from retrieval_grader import RetrievalGrader
from rag_chain import RetrievalAugmentedGenerator
from search_parser import SearchParser
from summarizer import Summarizer
from router import Router
from hallucination_grader import HallucinationGrader
from answer_grader import AnswerGrader
from ..api_clients.serp_api_client import SerpAPIClient
from langchain.schema import Document
from langgraph.graph import END, StateGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """
    question: str
    generation: str
    web_search: str
    documents: List[str]
    web_result: str


class GraphElements:

    def __init__(self, retriever: VectorStoreRetriever, config_path: str):
        self.retriever = retriever
        self.retrieval_grader = RetrievalGrader(config_path=config_path)
        self.rag_pipeline = RetrievalAugmentedGenerator(
            retriever=retriever,
            config_path=config_path
        )
        self.search_parser = SearchParser(config_path=config_path)
        self.router = Router(config_path=config_path)
        self.summarizer = Summarizer(config_path=config_path)
        self.hallucination_grader = HallucinationGrader(
            config_path=config_path
        )
        self.answer_grader = AnswerGrader(
            config_path=config_path
        )

    def route_question(self, state: GraphState) -> str:
        logging.info("--- \\ --- Routing user's question --- \\ ---")
        question = state["question"]

        source = self.router.generate_response(
            question=question
        )

        if source['datasource'] == 'web_search':
            logging.info("--- \\ --- Route question to web search --- \\ ---")
            return "websearch"

        elif source['datasource'] == 'vectorstore':
            logging.info("--- \\ --- Route question to rag --- \\ ---")
            return "vectorstore"

    def retrieve(self, state: GraphState) -> GraphState:
        logging.info("--- \\ --- Retrieving --- \\ ---")
        question = state["question"]

        documents = self.rag_pipeline.retrieve_context(question)
        return {"documents": documents, "question": question}

    def grade_documents(self, state: GraphState) -> GraphState:
        logging.info("--- \\ --- Checking documents relevance to the question --- \\ ---")
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        web_search = "No"

        for document in documents:
            grader_response = self.retrieval_grader.generate_response(
                question=question, document=document
            )
            grade = grader_response['score']

            if grade.lower() == "yes":
                logging.info("--- \\ --- Grade: document relevant! --- \\ ---")
                filtered_docs.append(document)

            else:
                logging.info("--- \\ --- Grade: document not relevant! --- \\ ---")

        if len(filtered_docs) == 0:
            logging.info("--- \\ --- No relevant documents found! --- \\ ---")
            web_search = "Yes"

        return {"documents": filtered_docs, "question": question, "web_search": web_search}

    def web_search(self, state: GraphState) -> GraphState:
        logging.info("--- \\ --- Web searching --- \\ ---")
        question = state["question"]

        try:
            documents = state["documents"]

        except KeyError:
            documents = None

        # web search
        response = SerpAPIClient().search_tool(query=question)
        search_parser_response = self.search_parser.generate_response(
            question=question,
            context=response.keys()
        )
        field = search_parser_response["field"]

        # # process output
        summarizer_response = self.summarizer.generate_response(
            question=question,
            context=response[field]
        )

        # Web search
        web_results = Document(page_content=summarizer_response)
        if documents is not None:
            documents.append(web_results)
        else:
            documents = [web_results]
        return {"documents": documents, "question": question, "web_result": summarized_data}

    def decide_to_generate(self, state: GraphState) -> str:
        logging.info("--- \\ --- Assessing graded documents --- \\ ---")
        web_search = state["web_search"]

        if web_search == "Yes":
            logging.info("--- \\ --- Decision: web search --- \\ ---")
            return "websearch"
        else:
            logging.info("--- \\ --- Decision: generate --- \\ ---")
            return "generate"

    def grade_generation(self, state: GraphState) -> str:
        logging.info("--- \\ --- Grading generation --- \\ ---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        web_search = state["web_search"]

        if web_search.lower() == "no":
            logging.info("--- \\ --- Checking hallucination --- \\ ---")
            hallucination_grader_response = self.hallucination_grader.generate_response(
                documents=documents, generation=generation
            )
            hallucination_grade = hallucination_grader_response['score']

            if hallucination_grade == "yes":
                logging.info("--- \\ --- Decision: generation is based on the documents --- \\ ---")
            else:
                logging.info("--- \\ --- Decision: generation is not based on the documents --- \\ ---")
                return "not supported"

            logging.info("--- \\ --- Checking answer --- \\ ---")
            answer_grader_response = self.answer_grader.generate_response(
                generation=generation, question=question
            )
            answer_grade = answer_grader_response['score']

            if answer_grade == "yes":
                logging.info("--- \\ --- Decision: generation addresses user's question --- \\ ---")
                return "useful"
            else:
                logging.info("--- \\ --- Decision: generation do not addresses user's question --- \\ ---")
                return "not useful"

    def generate(self, state: GraphState) -> GraphState:
        logging.info("--- \\ --- Generation --- \\ ---")
        question = state["question"]
        documents = state["documents"]
        web_result = state.get("web_result", None)

        if web_result is not None:
            return {"documents": documents, "question": question, "generation": web_result}

        else:
            rag_generation = self.rag_pipeline.generate_response(
                question=question,
                context=documents
            )
            return {"documents": documents, "question": question, "generation": rag_generation}

    def add_nodes(self) -> StateGraph:
        agent_graph = StateGraph(GraphState)

        agent_graph.add_node("web_search", self.web_search)
        agent_graph.add_node("retrieve", self.retrieve)
        agent_graph.add_node("grade_documents", self.grade_documents)
        agent_graph.add_node("generate", self.generate)

        return agent_graph

    def add_edges(self, agent_graph: StateGraph) -> StateGraph:
        agent_graph.set_conditional_entry_point(
            self.route_question,
            {
                "web_search": "web_search",
                "vectorstore": "retrieve",
            },
        )

        agent_graph.add_edge("retrieve", "grade_documents")
        agent_graph.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "web_search": "web_search",
                "generate": "generate",
            },
        )
        agent_graph.add_edge("web_search", "generate")
        agent_graph.add_conditional_edges(
            "generate",
            self.grade_generation,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "web_search",
            },
        )

        return agent_graph

    def build_graph(self) -> StateGraph:
        agent_graph = self.add_nodes()
        agent_graph = self.add_edges(agent_graph)
        compile_graph = agent_graph.compile_graph()
        return compile_graph
