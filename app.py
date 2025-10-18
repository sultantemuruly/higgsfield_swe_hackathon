from fastapi import FastAPI
from routers.text_to_image import router as text_to_image_router
from routers.text_to_video import router as text_to_video_router
from routers.image_to_video import router as image_to_video_router
from routers.ad_ideas import router as ad_ideas_router

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello, World!"}


app.include_router(
    text_to_image_router,
    prefix="/text-to-image",
    tags=["text-to-image"],
)

app.include_router(
    text_to_video_router,
    prefix="/text-to-video",
    tags=["text-to-video"],
)

app.include_router(
    image_to_video_router,
    prefix="/image_to_video_router",
    tags=["image_to_video_router"],
)

app.include_router(ad_ideas_router, prefix="/ad-ideas", tags=["ad-ideas"])
