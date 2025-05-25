import logging
import torch
from transformers import CLIPProcessor, CLIPModel
from .base_adapter import ModelServiceAdapter
from typing import Dict, Any
import asyncio
from PIL import Image
import io

logger = logging.getLogger(__name__)

class CLIPVisionAdapter(ModelServiceAdapter):
    """Adapter for CLIP vision model."""
    async def load(self, timeout: float = 1200.0) -> bool:
        """Load CLIP model and processor."""
        return await self.load_model(
            model_class=CLIPModel,
            tokenizer_class=lambda *args, **kwargs: CLIPProcessor.from_pretrained(
                *args,
                slow_image_processor_class=True,
                **kwargs
            ),
            timeout=timeout
        )

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle image feature extraction request."""
        try:
            request = await self.preprocess_input(request)
            image = Image.open(io.BytesIO(request['decoded_data']))
            inputs = self.tokenizer(images=image, return_tensors="pt").to(self.config['device'])
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            return await self.postprocess_output({'features': outputs.cpu().tolist()}, output_type='application/json')
        except Exception as e:
            return self._handle_error(e)