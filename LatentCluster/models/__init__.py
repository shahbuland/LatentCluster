from abc import abstractmethod

from torchtyping import TensorType

from torch import nn

class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.d_model = None

    @abstractmethod
    def preprocess(self, *inputs):
        """
        Preprocess whatever is being fetched from pipeline into forward inputs.
        """
        pass

    @abstractmethod
    def forward(self, *inputs) -> TensorType["batch", "d_model"]:
        """
        Embed input into latent state. Define how to extract latent state from model in subclasses.
        """
        pass

    def encode(self, *inputs):
        return self.forward(*self.preprocess(*inputs))