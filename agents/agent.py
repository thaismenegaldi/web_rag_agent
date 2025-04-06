from agents.graph_elements import GraphElements


class RunAgent:
    def __init__(self, retriever, config_path) -> None:
        self.compiled_graph = self.build_agent_graph(
            retriever=retriever, config_path=config_path
        )

    @staticmethod
    def build_agent_graph(retriever, config_path):
        graph_elements = GraphElements(
            retriever=retriever,
            config_path=config_path,
        )
        agent_graph = graph_elements.build_graph()
        return agent_graph

    def run_agent(self, question: str) -> str:
        inputs = {"question": question}
        for output in self.compiled_graph.stream(inputs):
            output = output.get("generate", None)
            if output is not None:
                output = output.get("generation", None)
        return output
