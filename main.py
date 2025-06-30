from fastapi import FastAPI, APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# Import API routers
from app.api import disease_detection, weed_detection, rag_chatbot

# Initialize FastAPI app
app = FastAPI(
    title="Crop Guard - Smart Agriculture System",
    description="Plant disease detection, weed detection, and AI assistant.",
    version="1.0.0"
)

# Enable CORS (allow frontend to communicate with backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# ---------------------------
# Web Routes (HTML Frontend)
# ---------------------------

web_router = APIRouter()

@web_router.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@web_router.get("/disease-detection", response_class=HTMLResponse)
async def disease_detection_page(request: Request):
    return templates.TemplateResponse("disease_detection.html", {"request": request})

@web_router.get("/weed-detection", response_class=HTMLResponse)
async def weed_detection_page(request: Request):
    return templates.TemplateResponse("weed_detection.html", {"request": request})

@web_router.get("/rag-chatbot", response_class=HTMLResponse)
async def rag_chatbot_page(request: Request):
    return templates.TemplateResponse("rag_chatbot.html", {"request": request})

# Include web routes
app.include_router(web_router)

# ---------------------------
# API Routes (Backends)
# ---------------------------

api_router = APIRouter(prefix="/api")

api_router.include_router(disease_detection.router, tags=["Disease Detection"])
api_router.include_router(weed_detection.router, tags=["Weed Detection"])
api_router.include_router(rag_chatbot.router, tags=["Chat Assistant"])

# Include API routes
app.include_router(api_router)
