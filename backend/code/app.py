# backend/code/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from query_system import QuerySystem
import os

app = FastAPI()

# CORS: Allow frontend running on localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class QueryRequest(BaseModel):
    state: str
    query: str

# Initialize system
data_path = os.path.join(os.getcwd(),'..','data', 'List_of_Schemes_Format_PM_10_B_2025_04_23_09_52.csv')
output_dir = os.path.join(os.getcwd(),'..', 'output')
gpt4all_model_path = "orca-mini-3b-gguf2-q4_0.gguf"
print(data_path)
query_system = QuerySystem(data_path, gpt4all_model_path, output_dir)

@app.post("/query")
def query_data(request: QueryRequest):
    if request.state.lower() != "all states":
        query_system.set_state_filter(request.state)
    else:
        query_system.set_state_filter(None)
    response = query_system.ask(request.query)
    return {"answer": response}
