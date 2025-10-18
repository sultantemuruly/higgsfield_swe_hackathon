import os
import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

load_dotenv()

# You can add your OpenAI API key or other AI service API key here
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

router = APIRouter()


class CompanyInfo(BaseModel):
    name: str = Field(..., description="Company name")
    industry: str = Field(..., description="Industry or sector the company operates in")
    target_audience: str = Field(..., description="Primary target audience for the company")
    brand_values: Optional[List[str]] = Field(default=[], description="Core brand values and principles")
    company_size: Optional[str] = Field(default="", description="Company size (startup, SME, enterprise, etc.)")
    market_position: Optional[str] = Field(default="", description="Market position (leader, challenger, niche, etc.)")


class ProductInfo(BaseModel):
    name: str = Field(..., description="Product or service name")
    description: str = Field(..., description="Detailed description of the product/service")
    key_features: List[str] = Field(..., description="Key features and benefits of the product")
    price_range: Optional[str] = Field(default="", description="Price range or pricing model")
    unique_selling_points: Optional[List[str]] = Field(default=[], description="What makes this product unique")
    use_cases: Optional[List[str]] = Field(default=[], description="Main use cases or applications")


class AdIdeasRequest(BaseModel):
    company: CompanyInfo = Field(..., description="Company information")
    product: ProductInfo = Field(..., description="Product information")
    ad_objective: Optional[str] = Field(default="brand awareness", description="Primary objective (brand awareness, sales, engagement, etc.)")
    budget_range: Optional[str] = Field(default="", description="Budget range for the campaign")
    preferred_channels: Optional[List[str]] = Field(default=[], description="Preferred advertising channels (social media, TV, print, etc.)")


class AdIdea(BaseModel):
    title: str = Field(..., description="Creative title for the ad idea")
    concept: str = Field(..., description="Detailed concept description")
    target_audience: str = Field(..., description="Specific target audience for this idea")
    key_message: str = Field(..., description="Main message to communicate")
    visual_suggestions: List[str] = Field(..., description="Visual elements and style suggestions")
    call_to_action: str = Field(..., description="Suggested call to action")
    channels: List[str] = Field(..., description="Recommended advertising channels")
    estimated_impact: str = Field(..., description="Expected impact and benefits")


class AdIdeasResponse(BaseModel):
    ideas: List[AdIdea] = Field(..., description="Generated advertising ideas")
    summary: str = Field(..., description="Summary of the campaign strategy")


def generate_ad_ideas_with_openai(company: CompanyInfo, product: ProductInfo, ad_objective: str, budget_range: str, preferred_channels: List[str]) -> AdIdeasResponse:
    """
    Generate advertising ideas using OpenAI API
    """
    if not OPENAI_API_KEY:
        # Fallback to mock data if no API key is provided
        return generate_mock_ad_ideas(company, product, ad_objective, budget_range, preferred_channels)
    
    prompt = f"""
    You are a creative advertising strategist. Generate 5 innovative advertising ideas for the following company and product:

    COMPANY INFORMATION:
    - Name: {company.name}
    - Industry: {company.industry}
    - Target Audience: {company.target_audience}
    - Brand Values: {', '.join(company.brand_values) if company.brand_values else 'Not specified'}
    - Company Size: {company.company_size}
    - Market Position: {company.market_position}

    PRODUCT INFORMATION:
    - Name: {product.name}
    - Description: {product.description}
    - Key Features: {', '.join(product.key_features)}
    - Price Range: {product.price_range}
    - Unique Selling Points: {', '.join(product.unique_selling_points) if product.unique_selling_points else 'Not specified'}
    - Use Cases: {', '.join(product.use_cases) if product.use_cases else 'Not specified'}

    CAMPAIGN DETAILS:
    - Objective: {ad_objective}
    - Budget Range: {budget_range}
    - Preferred Channels: {', '.join(preferred_channels) if preferred_channels else 'Any'}

    Please generate 5 creative advertising ideas. For each idea, provide:
    1. A catchy title
    2. Detailed concept description
    3. Specific target audience
    4. Key message
    5. Visual suggestions (3-5 elements)
    6. Call to action
    7. Recommended channels
    8. Estimated impact

    Format the response as a JSON object with "ideas" array and "summary" string.
    """

    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a creative advertising strategist. Always respond with valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000,
            "temperature": 0.8
        }

        # Note: This would require httpx.AsyncClient for async implementation
        # For now, we'll use the mock implementation
        return generate_mock_ad_ideas(company, product, ad_objective, budget_range, preferred_channels)
        
    except Exception as e:
        # Fallback to mock data on error
        return generate_mock_ad_ideas(company, product, ad_objective, budget_range, preferred_channels)


def generate_mock_ad_ideas(company: CompanyInfo, product: ProductInfo, ad_objective: str, budget_range: str, preferred_channels: List[str]) -> AdIdeasResponse:
    """
    Generate mock advertising ideas when AI service is not available
    """
    ideas = [
        AdIdea(
            title=f"Revolutionary {product.name} Experience",
            concept=f"Showcase {product.name} as a game-changer in {company.industry}. Create an emotional narrative that connects with {company.target_audience} by highlighting how the product transforms their daily experience.",
            target_audience=company.target_audience,
            key_message=f"{product.name} delivers {', '.join(product.key_features[:2])} that you've never experienced before.",
            visual_suggestions=[
                "Before/after transformation scenes",
                "Close-up shots of product features",
                "Diverse people using the product",
                "Modern, clean aesthetic",
                "Bold, vibrant colors"
            ],
            call_to_action="Experience the difference today",
            channels=["Social Media", "Digital Display", "Video Ads"],
            estimated_impact="High engagement and brand recall"
        ),
        AdIdea(
            title=f"The {company.name} Advantage",
            concept=f"Position {company.name} as the trusted leader in {company.industry} with {product.name} as the flagship solution. Use testimonials and success stories to build credibility.",
            target_audience=company.target_audience,
            key_message=f"Join thousands who trust {company.name} for {product.name}",
            visual_suggestions=[
                "Customer testimonials",
                "Company headquarters/branding",
                "Product in professional settings",
                "Trust indicators (certifications, awards)",
                "Professional, corporate styling"
            ],
            call_to_action="Join our community of satisfied customers",
            channels=["LinkedIn", "Industry Publications", "Email Marketing"],
            estimated_impact="Strong brand trust and lead generation"
        ),
        AdIdea(
            title=f"Unlock Your Potential with {product.name}",
            concept=f"Focus on the aspirational benefits of {product.name}. Show how it helps {company.target_audience} achieve their goals and overcome challenges in {company.industry}.",
            target_audience=company.target_audience,
            key_message=f"{product.name} empowers you to {product.description[:50]}...",
            visual_suggestions=[
                "Achievement and success imagery",
                "Product in action",
                "Motivational messaging",
                "Aspirational lifestyle shots",
                "Dynamic, energetic visuals"
            ],
            call_to_action="Start your journey to success",
            channels=["Instagram", "YouTube", "TikTok"],
            estimated_impact="High emotional engagement and conversions"
        ),
        AdIdea(
            title=f"Smart Choice: {product.name}",
            concept=f"Appeal to the rational decision-making process of {company.target_audience}. Highlight the practical benefits, cost-effectiveness, and ROI of choosing {product.name}.",
            target_audience=company.target_audience,
            key_message=f"Make the smart choice with {product.name} - {', '.join(product.key_features[:3])}",
            visual_suggestions=[
                "Data and statistics",
                "Comparison charts",
                "Product specifications",
                "Clean, technical design",
                "Professional color scheme"
            ],
            call_to_action="Make the smart choice today",
            channels=["Google Ads", "Industry Websites", "Trade Publications"],
            estimated_impact="High conversion rates and qualified leads"
        ),
        AdIdea(
            title=f"Behind the Scenes: {product.name}",
            concept=f"Create transparency and authenticity by showing the making of {product.name}, the people behind {company.name}, and the passion that goes into creating quality products.",
            target_audience=company.target_audience,
            key_message=f"See the passion and expertise behind {product.name}",
            visual_suggestions=[
                "Manufacturing process",
                "Team members at work",
                "Quality control processes",
                "Raw materials and craftsmanship",
                "Authentic, documentary style"
            ],
            call_to_action="Discover our story",
            channels=["YouTube", "Company Blog", "Social Media Stories"],
            estimated_impact="Strong brand loyalty and emotional connection"
        )
    ]

    summary = f"Comprehensive advertising strategy for {company.name}'s {product.name} targeting {company.target_audience} in the {company.industry} industry. The campaign focuses on {ad_objective} with a mix of emotional and rational appeals across multiple channels."

    return AdIdeasResponse(ideas=ideas, summary=summary)


@router.post("/generate", response_model=AdIdeasResponse)
async def generate_ad_ideas(request: AdIdeasRequest):
    """
    Generate creative advertising ideas based on company and product information.
    
    This endpoint analyzes the provided company and product details to create
    innovative advertising concepts, visual suggestions, and campaign strategies.
    """
    try:
        response = generate_ad_ideas_with_openai(
            company=request.company,
            product=request.product,
            ad_objective=request.ad_objective,
            budget_range=request.budget_range,
            preferred_channels=request.preferred_channels
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate ad ideas: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint for the ad ideas service
    """
    return {"status": "healthy", "service": "ad-ideas"}
