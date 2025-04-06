import logging
from typing import List

from langchain.schema import Document
from langchain_core.vectorstores.base import VectorStoreRetriever
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from agents.answer_grader import AnswerGrader
from agents.hallucination_grader import HallucinationGrader
from agents.rag_chain import RetrievalAugmentedGenerator
from agents.retrieval_grader import RetrievalGrader
from agents.router import Router
from agents.search_parser import SearchParser
from agents.summarizer import Summarizer
from api_clients.serp_api_client import SerpAPIClient
from utils.log_agent import log_agent_step

logging.basicConfig(
    level=logging.INFO,
    format="\n%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class GraphState(TypedDict):
    """
    Represents the state of the graph.

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
    retry_count: int


class GraphElements:

    def __init__(self, retriever: VectorStoreRetriever, config_path: str):
        self.retriever = retriever
        self.retrieval_grader = RetrievalGrader(config_path=config_path)
        self.rag_pipeline = RetrievalAugmentedGenerator(
            retriever=retriever, config_path=config_path
        )
        self.search_parser = SearchParser(config_path=config_path)
        self.router = Router(config_path=config_path)
        self.summarizer = Summarizer(config_path=config_path)
        self.hallucination_grader = HallucinationGrader(
            config_path=config_path
        )
        self.answer_grader = AnswerGrader(config_path=config_path)

    def route_question(self, state: GraphState) -> str:
        log_agent_step("Route user's question")
        question = state["question"]

        router_response = self.router.generate_response(question=question)

        if router_response.datasource == "web_search":
            logging.info("Route question to web search")
            return "search_in_web"

        elif router_response.datasource == "vector_store":
            logging.info("Route question to rag")
            return "vector_store"

    def retrieve(self, state: GraphState) -> GraphState:
        log_agent_step("Retrieve")
        question = state["question"]

        documents = self.rag_pipeline.retrieve_context(question)
        return {"documents": documents, "question": question}

    def grade_documents(self, state: GraphState) -> GraphState:
        logging.info("Checking documents relevance to the question")
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        web_search = "No"

        for document in documents:
            grader_response = self.retrieval_grader.generate_response(
                question=question, document=document
            )
            grade = grader_response.score

            if grade.lower() == "yes":
                logging.info("Grade: document relevant!")
                filtered_docs.append(document)

            else:
                logging.info("Grade: document not relevant!")

        if len(filtered_docs) == 0:
            logging.info("No relevant documents found!")
            web_search = "Yes"

        return {
            "documents": filtered_docs,
            "question": question,
            "web_search": web_search,
        }

    def web_search(self, state: GraphState) -> GraphState:
        log_agent_step("Web search")
        question = state["question"]
        retry_count = state.get("retry_count", 0)

        try:
            documents = state["documents"]

        except KeyError:
            documents = None

        if retry_count < 2:
            # web search
            response = SerpAPIClient().search_tool(query=question)
            search_parser_response = self.search_parser.generate_response(
                question=question, context=response.keys()
            )
            field = search_parser_response.field

            retry_count += 1

            # # process output
            summarizer_response = self.summarizer.generate_response(
                question=question, context=response[field]
            )
        else:
            summarizer_response = "Reached max retries."

        # Web search
        web_results = Document(page_content=summarizer_response)
        documents = [web_results]
        return {
            "documents": documents,
            "question": question,
            "web_result": summarizer_response,
            "retry_count": retry_count,
        }

    def decide_to_generate(self, state: GraphState) -> str:
        logging.info("Assessing graded documents")
        web_search = state["web_search"]

        if web_search == "Yes":
            logging.info("Decision: web search")
            return "search_in_web"
        else:
            logging.info("Decision: generate")
            return "generate"

    def grade_generation(self, state: GraphState) -> str:
        logging.info("Grading generation")

        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        web_result = state.get("web_result", None)

        if web_result == "Reached max retries.":
            return "not supported"

        elif web_result is None:
            logging.info("Checking hallucination")
            hallucination_grader_response = (
                self.hallucination_grader.generate_response(
                    documents=documents, generation=generation
                )
            )
            hallucination_grade = hallucination_grader_response.score

            if hallucination_grade == "yes":
                logging.info("Decision: generation is based on the documents")
            else:
                logging.info(
                    "Decision: generation is not based on the documents"
                )
                return "not supported"

        logging.info("Checking generation with user's question")
        answer_grader_response = self.answer_grader.generate_response(
            generation=generation, question=question
        )
        answer_grade = answer_grader_response.score

        if answer_grade == "yes":
            logging.info("Decision: generation addresses user's question")
            return "useful"
        else:
            logging.info(
                "Decision: generation do not addresses user's question"
            )
            return "not useful"

    def generate(self, state: GraphState) -> GraphState:
        log_agent_step("Generate")
        question = state["question"]
        documents = state["documents"]
        web_result = state.get("web_result", None)

        if web_result is not None:
            return {
                "documents": documents,
                "question": question,
                "generation": web_result,
            }

        else:
            rag_generation = self.rag_pipeline.generate_response(
                question=question, context=documents
            )
            return {
                "documents": documents,
                "question": question,
                "generation": rag_generation,
            }

    def add_nodes(self) -> StateGraph:
        agent_graph = StateGraph(GraphState)

        agent_graph.add_node("search_in_web", self.web_search)
        agent_graph.add_node("retrieve", self.retrieve)
        agent_graph.add_node("judge_context", self.grade_documents)
        agent_graph.add_node("generate", self.generate)

        return agent_graph

    def add_edges(self, agent_graph: StateGraph) -> StateGraph:
        agent_graph.set_conditional_entry_point(
            self.route_question,
            {
                "search_in_web": "search_in_web",
                "vector_store": "retrieve",
            },
        )

        agent_graph.add_edge("retrieve", "judge_context")
        agent_graph.add_conditional_edges(
            "judge_context",
            self.decide_to_generate,
            {
                "search_in_web": "search_in_web",
                "generate": "generate",
            },
        )
        agent_graph.add_edge("search_in_web", "generate")
        agent_graph.add_conditional_edges(
            "generate",
            self.grade_generation,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "search_in_web",
            },
        )

        return agent_graph

    def build_graph(self) -> StateGraph:
        agent_graph = self.add_nodes()
        agent_graph = self.add_edges(agent_graph)
        compile_graph = agent_graph.compile()
        return compile_graph
