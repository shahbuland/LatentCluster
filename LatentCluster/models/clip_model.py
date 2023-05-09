from typing import Union, Iterable

from LatentCluster.models import Encoder

from transformers import CLIPModel, CLIPProcessor, CLIPConfig 
from PIL import Image
import torch
from torch import nn

class CLIPEncoder(Encoder):
    """
    CLIP encoding model

    :param mode: TEXT or IMAGE, to determine if it will be used as a text or image encoder
    :type mode: str

    :param path: Path to a model on HF HUB that can be used to instantiate CLIP model
    :type path: str

    :param no_grad: Ensures all model forward calls use no_grad. Disable if you are interested in fine-tuning
    :type no_grad: bool
    """
    def __init__(self, mode : str, path : str, no_grad : bool = True):
        super().__init__()
    
        self.mode = mode
        self.path = path
        self.no_grad = no_grad

        self.config = CLIPConfig.from_pretrained(path)
        self.model = CLIPModel.from_pretrained(path)
        processor = CLIPProcessor.from_pretrained(path)
        self.tokenizer = processor.tokenizer
        self.image_processor = processor.image_processor

        self.d_model = self.model.projection_dim
        self.sequence_length = self.model.config.text_config.max_position_embeddings

    def preprocess(self, x  : Union[Iterable[Image.Image], Iterable[str]]) -> torch.Tensor:
        if self.mode == "TEXT":
            text = x
            features = self.tokenizer(
                text,
                return_tensors = 'pt', padding = 'max_length',
                truncation = True, max_length = self.sequence_length
            )
            return features['input_ids'], features['attention_mask']

        else:
            images = x
            features = self.image_processor(
                images,
                return_tensors = 'pt'
            )
            return [features['pixel_values']]

    def forward(self, *inputs):
        if self.no_grad:
            with torch.no_grad():
                return self._forward(*inputs)
        else:
            return self._forward(*inputs)

    def _forward(self, *inputs):
        if self.mode == "TEXT":
            input_ids, attention_mask = inputs
            res = self.model.get_text_features(input_ids, attention_mask)
        else:
            pixel_values = inputs
            res = self.model.get_image_features(pixel_values)

        return res