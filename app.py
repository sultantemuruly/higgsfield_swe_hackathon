from fastapi import FastAPI
from routers.text_to_image import router as text_to_image_router
from routers.text_to_video import router as text_to_video_router

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
