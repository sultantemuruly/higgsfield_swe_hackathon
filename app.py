from fastapi import FastAPI
from routers.text_to_image import router as text_to_image_router
from routers.ad_ideas import router as ad_ideas_router

app = FastAPI()

app.include_router(
    text_to_image_router, prefix="/text-to-image", tags=["text-to-image"]
)

app.include_router(
    ad_ideas_router, prefix="/ad-ideas", tags=["ad-ideas"]
)
