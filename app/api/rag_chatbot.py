from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.chat2 import setup_retrieval_qa
from app.services.chat1 import embedding_function
from langchain_chroma import Chroma

# Load vector database from disk
db = Chroma(persist_directory="./Agriculture_db", embedding_function=embedding_function)

# Initialize the RAG chain
chain = setup_retrieval_qa(db)

# Define request body schema
class ChatRequest(BaseModel):
    user_input: str

# Create router
router = APIRouter()

# Define POST endpoint for chatbot
@router.post("/chatbot")
async def chatbot_response(request: ChatRequest):
    query = request.user_input.strip().lower()

    # Handle special commands manually
    if query in ["exit", "q"]:
        return {"response": "Goodbye 👋"}

    if query in ["who developed you", "who created you", "who made you"]:
        return {"response": "I was developed by the CropPilot Team ❤️"}

    try:
        # Run the chain and get response
        response = chain.invoke(query)

        # Return result in a format expected by the frontend
        return {"response": response["result"]}

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Error while processing the question.")