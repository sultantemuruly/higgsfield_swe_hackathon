import os
import asyncio
import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, HttpUrl
from typing import Literal

load_dotenv()

HIGGSFIELD_API_KEY_ID = os.getenv("HIGGSFIELD_API_KEY_ID")
HIGGSFIELD_API_KEY_SECRET = os.getenv("HIGGSFIELD_API_KEY_SECRET")

if not HIGGSFIELD_API_KEY_ID or not HIGGSFIELD_API_KEY_SECRET:
    raise RuntimeError("Missing Credentials")

# separate router for image->video
router = APIRouter(prefix="/image-to-video", tags=["image-to-video"])


# You can delete these models if you import shared ones from elsewhere
class InputImage(BaseModel):
    type: Literal["image_url"] = Field(
        default="image_url",
        description="Image source type. Currently supports only 'image_url'.",
    )
    image_url: HttpUrl = Field(..., description="Publicly accessible image URL")


class ImageToVideoParams(BaseModel):
    prompt: str = Field(..., description="Text prompt")
    duration: int | float = Field(5, description="Video length in seconds")
    resolution: str = Field("720p", description="e.g., '720p', '1080p'")
    input_image: InputImage = Field(..., description="Single driving image")
    seed: int = Field(-1, description="Random seed; -1 for random")
    input_audio: HttpUrl | None = Field(None, description="Optional audio URL")
    enhance_prompt: bool = Field(False, description="Whether to enhance the prompt")
    negative_prompt: str = Field("", description="Negative prompt text")


def higgsfield_headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "hf-api-key": HIGGSFIELD_API_KEY_ID,
        "hf-secret": HIGGSFIELD_API_KEY_SECRET,
    }


@router.post("/wan-25-fast")
async def start_image_to_video(params: ImageToVideoParams):
    """
    Starts an image->video job on Higgsfield WAN-25-Fast.
    Returns the job-set payload containing the job_set_id.
    """
    url = "https://platform.higgsfield.ai/generate/wan-25-fast"
    payload = {"params": jsonable_encoder(params, exclude_none=True)}
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=higgsfield_headers(), json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _pick_video_url(data: dict) -> str | None:
    """
    Extract a playable video URL from a Higgsfield job-set payload.
    Tries several common shapes: results.url, results.video.url, results.raw.url, list items, etc.
    """

    def from_obj(obj: dict) -> str | None:
        # direct hits
        for key in ("url", "mp4", "video"):
            v = obj.get(key)
            if isinstance(v, str) and v.startswith("http"):
                return v
            if isinstance(v, dict):
                u = v.get("url")
                if isinstance(u, str) and u.startswith("http"):
                    return u

        # nested shapes like {"raw": {"url": ...}}, {"min": {"url": ...}}
        for nested in ("raw", "min", "high", "low"):
            v = obj.get(nested)
            if isinstance(v, dict):
                u = v.get("url")
                if isinstance(u, str) and u.startswith("http"):
                    return u
        return None

    # check top-level results
    top = data.get("results")
    if isinstance(top, dict):
        url = from_obj(top)
        if url:
            return url
    if isinstance(top, list):
        for item in top:
            if isinstance(item, dict):
                url = from_obj(item)
                if url:
                    return url

    # check per-job results
    for j in data.get("jobs", []):
        res = j.get("results")
        if not res:
            continue
        if isinstance(res, dict):
            url = from_obj(res)
            if url:
                return url
        if isinstance(res, list):
            for item in res:
                if isinstance(item, dict):
                    url = from_obj(item)
                    if url:
                        return url
    return None


@router.get("/jobs/{job_set_id}/wait-video")
async def wait_and_redirect_video(
    job_set_id: str, timeout: int = 180, interval: float = 3.0
):
    """
    Polls Higgsfield until the job-set is completed (<= timeout seconds),
    then 307-redirects to the video URL.
    """
    status_url = f"https://platform.higgsfield.ai/v1/job-sets/{job_set_id}"

    async with httpx.AsyncClient(timeout=30) as client:
        attempts = max(1, int(timeout / interval))
        for _ in range(attempts):
            r = await client.get(status_url, headers=higgsfield_headers())
            r.raise_for_status()
            data = r.json()

            vid_url = _pick_video_url(data)
            if vid_url:
                return RedirectResponse(url=vid_url, status_code=307)

            # surface job failures early
            for j in data.get("jobs", []):
                if j.get("status") in {"failed", "error"}:
                    raise HTTPException(status_code=502, detail="Generation failed")

            await asyncio.sleep(interval)

    raise HTTPException(
        status_code=202,
        detail="Video not ready yet. Keep polling or increase timeout.",
    )
