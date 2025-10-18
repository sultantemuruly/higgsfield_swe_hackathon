import os
import io
import asyncio
import httpx
from dotenv import load_dotenv
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field, HttpUrl

# ---------- env ----------
load_dotenv()
HIGGSFIELD_API_KEY_ID = os.getenv("HIGGSFIELD_API_KEY_ID")
HIGGSFIELD_API_KEY_SECRET = os.getenv("HIGGSFIELD_API_KEY_SECRET")
if not HIGGSFIELD_API_KEY_ID or not HIGGSFIELD_API_KEY_SECRET:
    raise RuntimeError("Missing Credentials")

# ---------- router ----------
router = APIRouter(prefix="/image-to-video", tags=["image-to-video"])


# ---------- models ----------
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
    # allow empty string for swagger examples; upstream usually tolerates ""
    input_audio: str | HttpUrl | None = Field(None, description="Optional audio URL")
    enhance_prompt: bool = Field(False, description="Whether to enhance the prompt")
    negative_prompt: str = Field("", description="Negative prompt text")


class GenerateITVRequest(BaseModel):
    model: Literal["wan-25-fast"]  # extend as needed
    params: ImageToVideoParams


# ---------- helpers ----------
def higgsfield_headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "hf-api-key": HIGGSFIELD_API_KEY_ID,
        "hf-secret": HIGGSFIELD_API_KEY_SECRET,
    }


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

    # top-level results
    top = data.get("results")
    if isinstance(top, dict):
        u = from_obj(top)
        if u:
            return u
    if isinstance(top, list):
        for item in top:
            if isinstance(item, dict):
                u = from_obj(item)
                if u:
                    return u

    # per-job results
    for j in data.get("jobs", []):
        res = j.get("results")
        if not res:
            continue
        if isinstance(res, dict):
            u = from_obj(res)
            if u:
                return u
        if isinstance(res, list):
            for item in res:
                if isinstance(item, dict):
                    u = from_obj(item)
                    if u:
                        return u
    return None


async def _poll_video_url(
    client: httpx.AsyncClient,
    job_set_id: str,
    timeout: int = 180,
    interval: float = 3.0,
) -> str | None:
    """
    Poll job-set until a video URL appears or timeout is reached.
    Raises HTTPException on upstream failure.
    """
    status_url = f"https://platform.higgsfield.ai/v1/job-sets/{job_set_id}"
    attempts = max(1, int(timeout / interval))

    for _ in range(attempts):
        r = await client.get(status_url, headers=higgsfield_headers())
        r.raise_for_status()
        data = r.json()

        vid_url = _pick_video_url(data)
        if vid_url:
            return vid_url

        # surface job failures early
        for j in data.get("jobs", []):
            if j.get("status") in {"failed", "error"}:
                raise HTTPException(
                    status_code=502, detail=j.get("error") or "Generation failed"
                )

        await asyncio.sleep(interval)

    return None


# ---------- legacy start endpoint (optional) ----------
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


# ---------- legacy poll+redirect (optional; not Swagger-friendly) ----------
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

            for j in data.get("jobs", []):
                if j.get("status") in {"failed", "error"}:
                    raise HTTPException(status_code=502, detail="Generation failed")

            await asyncio.sleep(interval)

    raise HTTPException(
        status_code=202, detail="Video not ready yet. Keep polling or increase timeout."
    )


# ---------- unified JSON endpoint (one-call, no redirects) ----------
@router.post("/generate")
async def generate_itv(
    body: GenerateITVRequest,
    timeout: int = 180,
    interval: float = 3.0,
):
    """
    Unified image-to-video (one call):
      1) Submit to /generate/{model}
      2) Poll /v1/job-sets/{id} until a video URL appears or timeout
      3) Return JSON with video_url and job_set_id (no redirects)
    """
    submit_url = f"https://platform.higgsfield.ai/generate/{body.model}"
    payload = {"params": jsonable_encoder(body.params, exclude_none=True)}

    job_set_id = None
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            submit = await client.post(
                submit_url, headers=higgsfield_headers(), json=payload
            )
            submit.raise_for_status()
            job_set = submit.json()
            job_set_id = job_set.get("id") or (job_set.get("data") or {}).get("id")
            if not job_set_id:
                raise HTTPException(status_code=502, detail="No job_set id in response")

            vid_url = await _poll_video_url(client, job_set_id, timeout, interval)
            if vid_url:
                return {
                    "status": "ready",
                    "model": body.model,
                    "job_set_id": job_set_id,
                    "video_url": vid_url,
                }

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # still processing
    raise HTTPException(
        status_code=202,
        detail={
            "message": "Video not ready yet. Increase timeout or try again.",
            "job_set_id": job_set_id,
            "model": body.model,
        },
    )


# ---------- unified BYTES endpoint (Swagger-friendly) ----------
@router.post(
    "/generate/bytes",
    responses={
        200: {"content": {"video/*": {}}, "description": "Generated video"},
        202: {"description": "Generation still in progress"},
        502: {"description": "Upstream generation failed"},
    },
)
async def generate_itv_bytes(
    body: GenerateITVRequest,
    timeout: int = 180,
    interval: float = 3.0,
):
    """
    Unified image-to-video with byte streaming:
      1) Submit to /generate/{model}
      2) Poll for the final video URL
      3) Server-side fetch & stream back video/* (no cross-origin redirects; Swagger-safe)
    """
    submit_url = f"https://platform.higgsfield.ai/generate/{body.model}"
    payload = {"params": jsonable_encoder(body.params, exclude_none=True)}

    job_set_id = None
    try:
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            submit = await client.post(
                submit_url, headers=higgsfield_headers(), json=payload
            )
            submit.raise_for_status()
            job_set = submit.json()
            job_set_id = job_set.get("id") or (job_set.get("data") or {}).get("id")
            if not job_set_id:
                raise HTTPException(status_code=502, detail="No job_set id in response")

            vid_url = await _poll_video_url(client, job_set_id, timeout, interval)
            if not vid_url:
                raise HTTPException(
                    status_code=202,
                    detail={
                        "message": "Video not ready yet. Try again or increase timeout.",
                        "job_set_id": job_set_id,
                        "model": body.model,
                    },
                )

            proxied = await client.get(vid_url, follow_redirects=True)
            proxied.raise_for_status()
            media = proxied.headers.get("content-type") or "video/mp4"
            return StreamingResponse(
                io.BytesIO(proxied.content),
                media_type=media if media.startswith("video/") else "video/mp4",
                headers={
                    "Cache-Control": "public, max-age=31536000",
                    "Content-Disposition": 'inline; filename="generated-video"',
                },
            )

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
