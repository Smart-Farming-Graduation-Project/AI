from fastapi import APIRouter, UploadFile, File, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.services.disease_detection_service import process_image
import uuid

# Initialize router with prefix and tag
router = APIRouter(
    prefix="/disease-detection",
    tags=["Disease Detection"]
)

# Response schema
class DetectionResponse(BaseModel):
    session_id: str
    image_url: str
    diseases: list[str]
    treatment: str

# POST endpoint for disease detection
@router.post("/", response_model=DetectionResponse)
async def predict_disease(request: Request, file: UploadFile = File(...)):
    try:
        # Get or generate session ID
        session_id = request.headers.get("X-Session-ID", str(uuid.uuid4()))

        # Process image using YOLO + MobileNet pipeline
        result = await process_image(file)

        # Return structured response
        return DetectionResponse(
            session_id=session_id,
            image_url=result.get("image_url"),
            diseases=result.get("diseases"),
            treatment=result.get("treatment")
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
