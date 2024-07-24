from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFaceHub
import pinecone
from dotenv import load_dotenv
import os

class ChatBot():
    def __init__(self):
        load_dotenv()
        self.loader = TextLoader('./horoscope.txt')
        self.documents = self.loader.load()
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        self.docs = self.text_splitter.split_documents(self.documents)

        self.embeddings = HuggingFaceEmbeddings()

        pinecone.init(
            api_key=os.getenv('PINECONE_API_KEY'),
            environment='gcp-starter'
        )

        self.index_name = "langchain-demo"

        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(name=self.index_name, metric="cosine", dimension=768)
            self.docsearch = Pinecone.from_documents(self.docs, self.embeddings, index_name=self.index_name)
        else:
            self.docsearch = Pinecone.from_existing_index(self.index_name, self.embeddings)

        self.repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.llm = HuggingFaceHub(
            repo_id=self.repo_id,
            model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50},
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )

        from langchain import PromptTemplate

        self.template = """
        Welcome to Laitkor Consultancy Services, I am tying to provide answer for your queries about Devops.

        Context: {context}
        Question: {question}
        Answer:
        """

        self.prompt = PromptTemplate(template=self.template, input_variables=["context", "question"])

        from langchain.schema.runnable import RunnablePassthrough
        from langchain.schema.output_parser import StrOutputParser

        self.rag_chain = (
            {"context": self.docsearch.as_retriever(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
