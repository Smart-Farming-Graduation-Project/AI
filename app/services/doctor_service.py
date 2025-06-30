import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the model
try:
    model = ChatGroq(
        model="Gemma2-9b-It",
        groq_api_key=groq_api_key
    )
except Exception as e:
    print(f"[Doctor Init Error] Could not initialize Groq model: {e}")
    model = None

# Use a general JSON parser to handle dynamic disease sets
parser = JsonOutputParser()

# Prompt allowing flexible output
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert agricultural doctor. For each disease I give you, provide a short, clear treatment plan in English. Return a JSON object where each disease name is a key, and the value is the treatment."),
    ("user", "Diseases: {diseases}")
])

# Connect the prompt, model, and parser
chain = prompt | model | parser if model else None

# Async callable function to get treatment
async def get_treatment_text(diseases: str) -> str:
    try:
        if model is None or chain is None:
            return "Treatment service is currently unavailable."

        result = await chain.ainvoke({"diseases": diseases})

        if isinstance(result, dict):
            return "\n".join([f"{k}: {v}" for k, v in result.items()])

        return str(result)

    except Exception as e:
        return f"Error while fetching treatment: {str(e)}"
