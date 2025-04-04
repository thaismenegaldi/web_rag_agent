from glob import glob
from tqdm import tqdm
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from ..utils.load_config import load_yaml_config
from langchain_core.vectorstores.base import VectorStoreRetriever


class VectorRetriever:
    def __init__(self, path_to_data: Path, config_path: Path):
        self.path_to_data = path_to_data
        self.config = load_yaml_config(config_path)

    def load_data(self) -> VectorStoreRetriever:
        files = glob(f"{self.path_to_data}/*.pdf")

        progress_bar = tqdm(
            total=len(files), desc="Loading files..."
        )
        documents = []

        for data_path in files:
            loader = PyPDFLoader(data_path)
            documents.extend(loader.load())

            progress_bar.update(1)

        progress_bar.close()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["retriever"]["chunk_size"],
            chunk_overlap=self.config["retriever"]["chunk_overlap"]
        )
        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name=self.config["retriever"]["model"],
        )

        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local("faiss_index")

        retriever = vectorstore.as_retriever()

        return retriever
