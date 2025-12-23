"""
OCRProcessor - LLM-based OCR processing.

Uses local LLM vision models (e.g., OlmOCR) for text extraction from images.
"""

from openai import AsyncOpenAI
import os
import sys
from dotenv import load_dotenv
import logging

load_dotenv()


class OCRProcessor:
    """
    LLM-based OCR processor.
    
    Uses async OpenAI-compatible API to communicate with local vision models
    for text extraction from images.
    """
    
    def __init__(self, api_base: str = None, model: str = None):
        """
        Initialize OCR processor.
        
        Args:
            api_base: LLM API base URL (default: from LLM_API_BASE env var or localhost:1234)
            model: LLM model name (default: from LLM_MODEL env var or allenai/olmocr-2-7b)
        """
        base_url = api_base or os.getenv("LLM_API_BASE", "http://localhost:1234/v1")
        model_name = model or os.getenv("LLM_MODEL", "allenai/olmocr-2-7b")
        
        self.client = AsyncOpenAI(base_url=base_url, api_key="lm-studio")
        self.model = model_name

    async def perform_ocr(self, image_base64):
        """
        Sends the image to the local LLM for OCR.
        Returns a list of strings (lines) representing the text.
        """
        text = await self._transcribe(image_base64)
        if not text:
            return []
        
        # Split into lines
        return [line.strip() for line in text.split('\n') if line.strip()]

    async def _transcribe(self, image_base64):
        """
        Internal method to transcribe an image using the LLM.
        
        Args:
            image_base64: Base64-encoded image string
            
        Returns:
            Transcribed text string or empty string on error
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Transcribe the text in this image accurately. Preserve line breaks. Return only the plain text."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                },
                            },
                        ],
                    }
                ],
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.debug(f"DEBUG: Transcription error: {e}")
            return ""
