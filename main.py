from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Union
from uvicorn import run
import os
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import shutil
from chatwithdoc import user_input
from chatwithdoc import get_pdf_text
from chatwithdoc import get_text_chunks
from chatwithdoc import get_vector_store
from chatwithdoc import vectorize_data

# Create an instance of FastAPI
class DocReq(BaseModel):
    prompt: str

app = FastAPI()

# Allow requests from your React app's domain
origins = [
    "http://localhost",
    "http://localhost:3000",
    "*"
    # Add more origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Define a route with a path parameter
@app.get("/hello/{name}")
async def read_item(name: str):
    return {"Hello": name}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/upload/{session_id}")
async def upload_file(session_id: int, file: UploadFile = File(...)):
    UPLOAD_DIR = Path(f"data_{session_id}")
    # Create the upload directory if it doesn't exist
    if not UPLOAD_DIR.exists():
        UPLOAD_DIR.mkdir() 

    # Save the uploaded file
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        vectorize_data(session_id)

    files = os.listdir(UPLOAD_DIR) 
    return {"filename": file.filename, "files": files}

@app.delete("/delete-all-files/{session_id}")
async def delete_all_files(session_id: int):
    try:
        UPLOAD_DIR = Path(f"data_{session_id}")
        INDEX_DIR = Path(f"faiss_index_{session_id}")
        if UPLOAD_DIR.exists():
            # Delete the entire folder
            shutil.rmtree(UPLOAD_DIR)
        if INDEX_DIR.exists():
            shutil.rmtree(INDEX_DIR)
        
        return {"message": "All files deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete_file/{session_id}")
async def delete_file(filename: str, session_id: int):
    UPLOAD_DIR = Path(f"data_{session_id}")
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        vectorize_data(session_id)
        return {"message": f"File {filename} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/files/{session_id}")
async def get_files(session_id: int):
    UPLOAD_DIR = Path(f"data_{session_id}")
    files = []
    if UPLOAD_DIR.exists():
        files = os.listdir(UPLOAD_DIR)
    return {"files": files}

@app.get("/index/{session_id}")
async def index_doc(session_id: int):
    response = vectorize_data(session_id)
    return response
    # pdf_folder = Path(f"data_{session_id}")
    # if pdf_folder.exists():
    #     pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    #     print("document to extract:", pdf_files)
    #     if pdf_files:
    #         # extracted_text = get_pdf_text(pdf_files)
    #         # text_chunks = get_text_chunks(extracted_text)
    #         # get_vector_store(text_chunks, session_id)
    #         vectorize_data(pdf_files, session_id)
    #         return {"message": "Document vectorized successfully.."}
    # else :
    #     return {"message": "Failed to vectorize, Please check if document exists."}

@app.post("/process_doc/{session_id}")
async def process_doc(req: DocReq, session_id: int):
    try:
        response = user_input(req.prompt, session_id)
        return {"response": response}
    except Exception as e:
        print("Raise of exception:", str(e))
        raise HTTPException(status_code=404, detail=str(e))
    

   
# Run the server with custom port
if __name__ == "__main__":
    run(app, host="127.0.0.1", port=8000)