import os
import asyncio
import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

load_dotenv()

HIGGSFIELD_API_KEY_ID = os.getenv("HIGGSFIELD_API_KEY_ID")
HIGGSFIELD_API_KEY_SECRET = os.getenv("HIGGSFIELD_API_KEY_SECRET")

if not HIGGSFIELD_API_KEY_ID or not HIGGSFIELD_API_KEY_SECRET:
    raise RuntimeError("Missing Credentials")

router = APIRouter()


class ModelParams(BaseModel):
    prompt: str = Field(..., description="Text prompt")
    duration: int = Field(6, description="Clip length in seconds")
    resolution: str = Field("768", description="Video side length, e.g. 576, 720, 768")
    enable_prompt_optimizier: bool = Field(True, description="Enable prompt optimizer")


def higgsfield_headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "hf-api-key": HIGGSFIELD_API_KEY_ID,
        "hf-secret": HIGGSFIELD_API_KEY_SECRET,
    }


@router.post("/minimax-t2v")
async def generate_minimax_t2v(params: ModelParams):
    """
    Submit a text-to-video job to MiniMax T2V.
    Returns the job-set JSON (contains job_set_id you can poll).
    """
    url = "https://platform.higgsfield.ai/generate/minimax-t2v"
    payload = {"params": params.model_dump()}

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=higgsfield_headers(), json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/seedance-v1-lite-t2v")
async def generate_minimax_t2v(params: ModelParams):

    url = "https://platform.higgsfield.ai/generate/seedance-v1-lite-t2v"
    payload = {"params": params.model_dump()}

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
    Scan a job-set payload and return the first likely video URL.
    Tries common shapes: results.video.url, results.video_url, results.raw.url, results.url, etc.
    """

    def from_results(res: dict) -> str | None:
        # common keys for video-like results
        candidates = [
            (res.get("video") or {}).get("url"),
            res.get("video_url"),
            (res.get("raw") or {}).get("url"),
            (res.get("min") or {}).get("url"),
            res.get("url"),
        ]
        for u in candidates:
            if isinstance(u, str) and u.startswith("http"):
                # prefer video extensions but fallback to any URL
                if any(u.lower().endswith(ext) for ext in (".mp4", ".webm", ".mov")):
                    return u
        # fallback: first http-like string anywhere at top level
        for v in res.values():
            if isinstance(v, str) and v.startswith("http"):
                return v
        return None

    for j in data.get("jobs", []):
        res = j.get("results")
        if not res:
            continue
        if isinstance(res, dict):
            u = from_results(res)
            if u:
                return u
        elif isinstance(res, list):
            for item in res:
                if isinstance(item, dict):
                    u = from_results(item)
                    if u:
                        return u
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

            # bail out early on failure
            for j in data.get("jobs", []):
                if j.get("status") in {"failed", "error"}:
                    raise HTTPException(status_code=502, detail="Generation failed")

            await asyncio.sleep(interval)

    raise HTTPException(
        status_code=202,
        detail="Video not ready yet. Keep polling or increase timeout.",
    )
