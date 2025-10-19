import os
import io
import asyncio
from typing import Literal, Union, Annotated

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ---------- env ----------
load_dotenv()
HIGGSFIELD_API_KEY_ID = os.getenv("HIGGSFIELD_API_KEY_ID")
HIGGSFIELD_API_KEY_SECRET = os.getenv("HIGGSFIELD_API_KEY_SECRET")
if not HIGGSFIELD_API_KEY_ID or not HIGGSFIELD_API_KEY_SECRET:
    raise RuntimeError("Missing Credentials")

# ---------- router ----------
router = APIRouter(prefix="/text-to-video", tags=["text-to-video"])


# ---------- models ----------
class _BaseParams(BaseModel):
    prompt: str = Field(..., description="Text prompt")
    # keep the original upstream field name (misspelling) for compatibility
    enable_prompt_optimizier: bool = Field(True, description="Enable prompt optimizer")


# minimax-t2v: duration ∈ {6,10}, resolution ∈ {768,1080}
class MiniMaxParams(_BaseParams):
    duration: Literal[6, 10] = Field(6, description="Clip length in seconds (6 or 10)")
    resolution: Literal["768", "1080"] = Field(
        "768", description="Video side length (768 or 1080)"
    )


# seedance-v1-lite-t2v: duration ∈ {5,10}, resolution ∈ {468,720,1080}
class SeedanceParams(_BaseParams):
    duration: Literal[5, 10] = Field(5, description="Clip length in seconds (5 or 10)")
    resolution: Literal["468", "720", "1080"] = Field(
        "720", description="Video side length (468, 720, or 1080)"
    )


class _GenerateMini(BaseModel):
    model: Literal["minimax-t2v"]
    params: MiniMaxParams


class _GenerateSeed(BaseModel):
    model: Literal["seedance-v1-lite-t2v"]
    params: SeedanceParams


GenerateVideoRequest = Annotated[
    Union[_GenerateMini, _GenerateSeed],
    Field(discriminator="model"),
]


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
    Scan a job-set payload and return the first likely video URL.
    Tries common shapes: results.video.url, results.video_url, results.raw.url, results.url, etc.
    """

    def from_results(res: dict) -> str | None:
        candidates = [
            (res.get("video") or {}).get("url"),
            res.get("video_url"),
            (res.get("raw") or {}).get("url"),
            (res.get("min") or {}).get("url"),
            res.get("url"),
        ]
        for u in candidates:
            if isinstance(u, str) and u.startswith("http"):
                if any(
                    u.lower().endswith(ext) for ext in (".mp4", ".webm", ".mov", ".m4v")
                ):
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

        for j in data.get("jobs", []):
            if j.get("status") in {"failed", "error"}:
                raise HTTPException(
                    status_code=502, detail=j.get("error") or "Generation failed"
                )

        await asyncio.sleep(interval)

    return None


# ---------- optional legacy endpoints ----------
@router.post("/minimax-t2v")
async def generate_minimax_t2v(params: MiniMaxParams):
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
async def generate_seedance_v1_lite_t2v(params: SeedanceParams):
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


@router.get("/jobs/{job_set_id}/wait-video")
async def wait_and_redirect_video(
    job_set_id: str, timeout: int = 180, interval: float = 3.0
):
    """
    (Legacy) Polls Higgsfield until the job-set is completed (<= timeout seconds),
    then returns a 307 redirect to the video URL.
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
                # 307 preserves method; prefer using the unified endpoint below for Swagger-friendly flow.
                return StreamingResponse(
                    iter(()), status_code=307, headers={"Location": vid_url}
                )

            for j in data.get("jobs", []):
                if j.get("status") in {"failed", "error"}:
                    raise HTTPException(status_code=502, detail="Generation failed")
            await asyncio.sleep(interval)
    raise HTTPException(
        status_code=202,
        detail="Video not ready yet. Keep polling or increase timeout.",
    )


# ---------- unified endpoint (ONE CALL): submit -> poll -> return video ----------
@router.post(
    "/generate",  # NOTE: lives under /text-to-video/generate
    responses={
        200: {
            "content": {"video/*": {}, "application/json": {}},
            "description": "Generated video or JSON link",
        },
        202: {"description": "Generation still in progress"},
        502: {"description": "Upstream generation failed"},
    },
)
async def generate_video(
    body: GenerateVideoRequest,
    timeout: int = 180,
    interval: float = 3.0,
    mode: Literal["bytes", "json"] = "bytes",
):
    """
    Unified text-to-video:
      1) Submits to /generate/{model}
      2) Polls /v1/job-sets/{id} until a video URL appears or timeout
      3) Returns:
         - mode=bytes (default): streams the video bytes back with video/* (Swagger-friendly)
         - mode=json: returns JSON with video_url and job_set_id
    """
    submit_url = f"https://platform.higgsfield.ai/generate/{body.model}"
    payload = {"params": body.params.model_dump()}

    job_set_id = None
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            # 1) submit job
            submit = await client.post(
                submit_url, headers=higgsfield_headers(), json=payload
            )
            submit.raise_for_status()
            job_set = submit.json()
            job_set_id = job_set.get("id") or (job_set.get("data") or {}).get("id")
            if not job_set_id:
                raise HTTPException(status_code=502, detail="No job_set id in response")

            # 2) poll for video URL
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

            # 3) return chosen mode
            if mode == "json":
                return {
                    "status": "ready",
                    "model": body.model,
                    "job_set_id": job_set_id,
                    "video_url": vid_url,
                }

            # mode == "bytes": proxy the video and stream to caller (no CORS issues)
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
