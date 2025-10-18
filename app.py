import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import httpx
import asyncio
from fastapi.responses import RedirectResponse

load_dotenv()

HIGGSFIELD_API_KEY_ID = os.getenv("HIGGSFIELD_API_KEY_ID")
HIGGSFIELD_API_KEY_SECRET = os.getenv("HIGGSFIELD_API_KEY_SECRET")

if not HIGGSFIELD_API_KEY_ID or not HIGGSFIELD_API_KEY_SECRET:
    raise RuntimeError("Missing Credentials")

app = FastAPI()


class NanoBananaParams(BaseModel):
    prompt: str = Field(..., description="Text prompt")
    aspect_ratio: str = Field("4:3", description="e.g., '4:3', '1:1', '16:9'")
    input_images: list[str]


def higgsfield_headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "hf-api-key": HIGGSFIELD_API_KEY_ID,
        "hf-secret": HIGGSFIELD_API_KEY_SECRET,
    }


@app.post("/higgsfield/nano-banana")
async def generate_nano_banana(params: NanoBananaParams):
    url = "https://platform.higgsfield.ai/v1/text2image/nano-banana"
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


def _pick_image_url(data: dict) -> str | None:
    for j in data.get("jobs", []):
        res = j.get("results")
        if not res:
            continue
        if isinstance(res, dict):
            # prefer full-res 'raw', fallback to 'min'
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


@app.get("/higgsfield/jobs/{job_set_id}/wait-image")
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

            # if completed, grab URL and redirect
            img_url = _pick_image_url(data)
            if img_url:
                return RedirectResponse(url=img_url, status_code=307)

            # optional: quick failure check
            for j in data.get("jobs", []):
                if j.get("status") in {"failed", "error"}:
                    raise HTTPException(status_code=502, detail="Generation failed")

            await asyncio.sleep(interval)

    # still not ready
    raise HTTPException(
        status_code=202, detail="Image not ready yet. Keep polling or increase timeout."
    )
