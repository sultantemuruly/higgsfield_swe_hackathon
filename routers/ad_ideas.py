import os
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import google.generativeai as genai

load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

router = APIRouter(prefix="/ad_ideas", tags=["ad_ideas"])


class ProductInfo(BaseModel):
    name: str = Field(..., description="Product or service name")
    description: str = Field(..., description="Description of the product/service")


class AdIdeasRequest(BaseModel):
    product: ProductInfo = Field(..., description="Product information")
    company_words: Optional[str] = Field(
        None, description="Company-specific keywords/phrases"
    )


class AdIdeasResponse(BaseModel):
    text: str = Field(..., description="Generated advertising idea")


def generate_ad_ideas_with_gemini(
    product: ProductInfo, company_words: str = None
) -> AdIdeasResponse:
    """
    Generate one creative advertising idea using Gemini SDK.
    """
    model = genai.GenerativeModel("gemini-2.5-flash")

    company_context = (
        f" using these company values: {company_words}" if company_words else ""
    )
    prompt = f"""
    You are a creative advertising strategist. Generate 1 innovative advertising idea for this product:

    PRODUCT INFORMATION:
    - Name: {product.name}
    - Description: {product.description}
    {company_context}

    Generate 1 creative advertising idea. Provide a catchy title and detailed concept description (2–3 sentences).
    Return ONLY the advertising idea text, no JSON formatting.
    """

    try:
        response = model.generate_content(prompt)
        if not response or not response.text:
            raise ValueError("Empty response from Gemini")
        return AdIdeasResponse(text=response.text.strip())

    except Exception as e:
        print(f"Gemini SDK error: {e}")
        return generate_mock_ad_idea(product, company_words)


def generate_mock_ad_idea(
    product: ProductInfo, company_words: str = None
) -> AdIdeasResponse:
    """
    Fallback: Generate a simple advertising idea without Gemini.
    """
    company_context = f" using {company_words}" if company_words else ""
    idea_text = (
        f"Revolutionary {product.name} Experience: Showcase {product.name} as a game-changer{company_context}. "
        f"Create an emotional narrative that highlights how {product.description} transforms daily experiences "
        f"and delivers exceptional value to users."
    )
    return AdIdeasResponse(text=idea_text)


@router.post("/generate", response_model=AdIdeasResponse)
async def generate_ad_ideas(request: AdIdeasRequest):
    """
    Generate 1 creative advertising idea based on product info and company words.
    """
    try:
        return generate_ad_ideas_with_gemini(
            product=request.product, company_words=request.company_words
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate ad idea: {str(e)}"
        )


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ad-ideas"}
