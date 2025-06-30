from fastapi import APIRouter, UploadFile, File, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.services.weed_service import detect_weed
import uuid

# Router setup
router = APIRouter(
    prefix="/weed-detection",
    tags=["Weed Detection"]
)

# Response model
class WeedDetectionResponse(BaseModel):
    session_id: str
    image_url: str
    has_weed: bool
    confidence: float

@router.post("/", response_model=WeedDetectionResponse)
async def weed_detection_api(request: Request, file: UploadFile = File(...)):
    try:
        # Generate or get session ID
        session_id = request.headers.get("X-Session-ID", str(uuid.uuid4()))
        
        # Call YOLO detection
        result = await detect_weed(file)

        return WeedDetectionResponse(
            session_id=session_id,
            image_url=result.get("image_url"),
            has_weed=result.get("has_weed", False),
            confidence=result.get("confidence", 0.0)
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
