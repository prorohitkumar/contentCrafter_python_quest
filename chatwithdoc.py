import os
import google.generativeai as genai

from pathlib import Path

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, session_id):
    # Create a FAISS vector of data and store locally
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    save_index_to = f'faiss_index_{session_id}'
    vector_store.save_local(save_index_to)

def vectorize_data(session_id):

    pdf_folder = Path(f"data_{session_id}")
    if pdf_folder.exists():
        pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        print("document to extract:", pdf_files)
        if pdf_files:
            extracted_text = get_pdf_text(pdf_files)
            text_chunks = get_text_chunks(extracted_text)
            get_vector_store(text_chunks, session_id)
            # vectorize_data(pdf_files, session_id)
            return {"message": "Document vectorized successfully.."}
        else :
            return {"message": "Failed to vectorize, Please check if document exists."}

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question, session_id):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    # get_index_from = f'faiss_index_{session_id}'
    get_index_from = Path(f'faiss_index_{session_id}')
    if get_index_from.exists():
        new_db = FAISS.load_local(get_index_from, embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        print("Prompt:", user_question)
        chain = get_conversational_chain()

        response = chain(
            {"input_documents":docs, "question": user_question}
            , return_only_outputs=True)

        print(response)
        return response
    else:
        print("Index not found")
        response = {"output_text":"Failed to load document index, Please try again."}
        return response



# def main():
    # session_id = "010101"
    # pdf_folder = f'./data_{session_id}'  # Replace with the path to your PDF folder
    # pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    # print("document to extract:", pdf_files)
    # extracted_text = get_pdf_text(pdf_files)
    # text_chunks = get_text_chunks(extracted_text)
    # get_vector_store(text_chunks)
    # user_input("Encoder and Decoder Stacks")
    

# if __name__ == "__main__":
#     main()