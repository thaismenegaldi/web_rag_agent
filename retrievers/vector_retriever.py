import logging
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

from utils.load_config import load_yaml_config

logging.basicConfig(
    level=logging.INFO,
    format="\n%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class VectorRetriever:
    def __init__(self, path_to_data: Path, config_path: Path):
        self.path_to_data = path_to_data
        self.config = load_yaml_config(config_path)

    def load_data(self):
        data_path = Path(self.path_to_data)
        entries = [p.name for p in data_path.iterdir()]
        pdf_files = list(data_path.glob("*.pdf"))

        if "faiss_index" not in entries:
            progress_bar = tqdm(total=len(pdf_files), desc="Loading files...")
            documents = []

            for data_path in pdf_files:
                loader = PyPDFLoader(data_path)
                documents.extend(loader.load())

                progress_bar.update(1)

            progress_bar.close()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config["retriever"]["chunk_size"],
                chunk_overlap=self.config["retriever"]["chunk_overlap"],
            )
            texts = text_splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(
                model_name=self.config["retriever"]["model"],
            )

            vector_store = FAISS.from_documents(texts, embeddings)
            vector_store.save_local("faiss_index")

        else:
            logging.info("Loading local vector store...")
            embeddings = HuggingFaceEmbeddings(
                model_name=self.config["retriever"]["model"],
            )

            vector_store = FAISS.load_local(
                "faiss_index", embeddings, allow_dangerous_deserialization=True
            )

        retriever = vector_store.as_retriever()

        return retriever
