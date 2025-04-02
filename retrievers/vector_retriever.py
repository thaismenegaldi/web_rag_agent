from glob import glob
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class VectorRetriever:
    def __init__(self, path_to_data):
        self.path_to_data = path_to_data

    def load_data(self):
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
            chunk_size=500, chunk_overlap=50
        )
        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local("faiss_index")

        retriever = vectorstore.as_retriever()

        return retriever
