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
    raise RuntimeError(
        "Missing Credentials: set HIGGSFIELD_API_KEY_ID and HIGGSFIELD_API_KEY_SECRET"
    )

# ---------- router ----------
router = APIRouter(prefix="/text-to-image", tags=["text-to-image"])


# ---------- models ----------
class InputImage(BaseModel):
    type: Literal["image_url"] = Field(
        default="image_url",
        description="Image source type. Currently supports only 'image_url'.",
    )
    image_url: HttpUrl = Field(..., description="Publicly accessible image URL")


class ModelParams(BaseModel):
    prompt: str = Field(..., description="Text prompt")
    aspect_ratio: str = Field("4:3", description="e.g., '4:3', '1:1', '16:9'")
    input_images: list[InputImage] | None = Field(
        None, description="Optional list of input images"
    )


class GenerateImageRequest(BaseModel):
    model: Literal["nano-banana", "seedream"]
    params: ModelParams


# ---------- helpers ----------
def higgsfield_headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "hf-api-key": HIGGSFIELD_API_KEY_ID,
        "hf-secret": HIGGSFIELD_API_KEY_SECRET,
    }


def _pick_image_url(data: dict) -> str | None:
    """
    Extract first usable image URL from a Higgsfield job-set payload.
    Prefers full-res 'raw.url', then 'min.url', then 'url'.
    """
    for j in data.get("jobs", []):
        res = j.get("results")
        if not res:
            continue
        if isinstance(res, dict):
            return (
                (res.get("raw") or {}).get("url")
                or (res.get("min") or {}).get("url")
                or res.get("url")
            )
        if isinstance(res, list):
            for item in res:
                url = (
                    (item.get("raw") or {}).get("url")
                    or (item.get("min") or {}).get("url")
                    or item.get("url")
                )
                if url:
                    return url
    return None


async def _poll_image_url(
    client: httpx.AsyncClient,
    job_set_id: str,
    timeout: int = 75,
    interval: float = 2.0,
) -> str | None:
    """
    Poll job-set until an image URL appears or timeout is reached.
    Raises HTTPException on upstream failure.
    """
    status_url = f"https://platform.higgsfield.ai/v1/job-sets/{job_set_id}"
    attempts = max(1, int(timeout / interval))

    for _ in range(attempts):
        r = await client.get(status_url, headers=higgsfield_headers())
        r.raise_for_status()
        data = r.json()

        img_url = _pick_image_url(data)
        if img_url:
            return img_url

        for j in data.get("jobs", []):
            if j.get("status") in {"failed", "error"}:
                raise HTTPException(
                    status_code=502, detail=j.get("error") or "Generation failed"
                )

        await asyncio.sleep(interval)

    return None


# ---------- optional: model-specific endpoints ----------
@router.post("/nano-banana")
async def generate_nano_banana(params: ModelParams):
    url = "https://platform.higgsfield.ai/v1/text2image/nano-banana"
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


@router.post("/seedream")
async def generate_seedream(params: ModelParams):
    url = "https://platform.higgsfield.ai/v1/text2image/seedream"
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


# ---------- legacy poll+redirect (kept for completeness; not needed for Swagger) ----------
@router.get("/jobs/{job_set_id}/wait-image")
async def wait_and_redirect_image(
    job_set_id: str, timeout: int = 75, interval: float = 2.0
):
    """
    Polls Higgsfield until the job-set is completed (<= timeout seconds),
    then 307-redirects to the image URL.
    """
    status_url = f"https://platform.higgsfield.ai/v1/job-sets/{job_set_id}"
    async with httpx.AsyncClient(timeout=30) as client:
        attempts = max(1, int(timeout / interval))
        for _ in range(attempts):
            r = await client.get(status_url, headers=higgsfield_headers())
            r.raise_for_status()
            data = r.json()

            img_url = _pick_image_url(data)
            if img_url:
                return RedirectResponse(url=img_url, status_code=307)

            for j in data.get("jobs", []):
                if j.get("status") in {"failed", "error"}:
                    raise HTTPException(status_code=502, detail="Generation failed")

            await asyncio.sleep(interval)

    raise HTTPException(
        status_code=202, detail="Image not ready yet. Keep polling or increase timeout."
    )


# ---------- unified JSON endpoint (no redirects) ----------
@router.post("/generate")
async def generate_image(
    body: GenerateImageRequest,
    timeout: int = 75,
    interval: float = 2.0,
):
    """
    One-call image generation:
    - Submits to /v1/text2image/{model}
    - Polls /v1/job-sets/{id} until an image URL appears or timeout
    - Returns JSON with image_url and job_set_id (no redirects)
    """
    submit_url = f"https://platform.higgsfield.ai/v1/text2image/{body.model}"
    payload = {"params": jsonable_encoder(body.params, exclude_none=True)}

    job_set_id = None
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            submit = await client.post(
                submit_url, headers=higgsfield_headers(), json=payload
            )
            submit.raise_for_status()
            job_set = submit.json()
            job_set_id = job_set.get("id")
            if not job_set_id:
                raise HTTPException(status_code=502, detail="No job_set id in response")

            img_url = await _poll_image_url(client, job_set_id, timeout, interval)
            if img_url:
                return {
                    "status": "ready",
                    "model": body.model,
                    "job_set_id": job_set_id,
                    "image_url": img_url,
                }

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # timeout (still processing)
    raise HTTPException(
        status_code=202,
        detail={
            "message": "Image not ready yet. Increase timeout or try again.",
            "job_set_id": job_set_id,
            "model": body.model,
        },
    )


# ---------- unified BYTES endpoint (Swagger-friendly) ----------
@router.post(
    "/generate/bytes",
    responses={
        200: {"content": {"image/*": {}}, "description": "Generated image"},
        202: {"description": "Generation still in progress"},
        502: {"description": "Upstream generation failed"},
    },
)
async def generate_image_bytes(
    body: GenerateImageRequest,
    timeout: int = 75,
    interval: float = 2.0,
):
    """
    Submit -> poll -> fetch the image URL -> stream bytes back with image/* content-type.
    Works cleanly in Swagger UI (no cross-origin redirects).
    """
    submit_url = f"https://platform.higgsfield.ai/v1/text2image/{body.model}"
    payload = {"params": jsonable_encoder(body.params, exclude_none=True)}

    job_set_id = None
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            # 1) submit
            submit = await client.post(
                submit_url, headers=higgsfield_headers(), json=payload
            )
            submit.raise_for_status()
            job_set = submit.json()
            job_set_id = job_set.get("id")
            if not job_set_id:
                raise HTTPException(status_code=502, detail="No job_set id in response")

            # 2) poll
            img_url = await _poll_image_url(client, job_set_id, timeout, interval)
            if not img_url:
                raise HTTPException(
                    status_code=202,
                    detail={
                        "message": "Image not ready yet. Try again or increase timeout.",
                        "job_set_id": job_set_id,
                        "model": body.model,
                    },
                )

            # 3) fetch image and stream back
            img_res = await client.get(img_url, follow_redirects=True)
            img_res.raise_for_status()
            media = img_res.headers.get("content-type", "image/jpeg")
            return StreamingResponse(
                io.BytesIO(img_res.content),
                media_type=media,
                headers={
                    "Cache-Control": "public, max-age=31536000",
                    "Content-Disposition": 'inline; filename="generated-image"',
                },
            )

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
