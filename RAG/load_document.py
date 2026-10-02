from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
 
#loader = PyPDFLoader("Deplyoment_and_Configuring_Autoscaling_for_Stateless_Applicaitinos.pdf")

pdf_file = [
    "Deployment.pdf", "another_document.pdf", "doccker.pdf"
]

documents = []
for pdf_file in pdf_file:
    documents.extend(PyPDFLoader(pdf_file).load())
    
    
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

chunks = text_splitter.split_documents(documents)
print(f"Total document chunks created: {len(chunks)}")