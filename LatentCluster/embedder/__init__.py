from typing import Iterable, Any

from LatentCluster.models import Encoder
from LatentCluster.utils import ProgressSampler

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
import joblib
from tqdm import tqdm

class BaseEmbedder:
    """
    Base class to embed large dataset using model. Result is added to an empty array

    :param model: Model to use for encoding
    
    :param chunk_size: Size to chunk input when running through model

    :param dataset: Dataset that will be iterated through

    :param precision: Float precision for embeddings

    """
    def __init__(self, model : Encoder, dataset : Iterable[any], 
        chunk_size : int = 128,
        precision : str = "fp16",
        device : str = "cuda",
        save_every: int = 64,
        fp : str = "embed.joblib"
    ):
        self.model = model
        self.model.to(device)

        self.chunk_size = chunk_size
        self.dataset = dataset
        self.precision = precision
        self.save_every = save_every
        self.fp = fp
        self.device = device

        self.precision_types = {
            "fp16" : np.float16,
            "fp32" : np.float32
        }

        self.res_arr = np.empty((len(self.dataset), self.model.d_model), dtype = self.precision_types[precision])
        self.progress = 0

    def get_embedding():
        return self.res_arr[:self.progress]
    
    def postprocess(self, x : torch.Tensor) -> Any:
        """
        Process model outputs into something that can be added to self.res_arr
        """

        if type(x) is torch.Tensor:
            if self.precision == "fp16":
                x = x.half()
            
            return x.cpu().numpy()
        elif type(x) is np.ndarray:
            x = np.astype(self.precision_types[self.precision])
        else:
            raise ValueError("Error: Model output cannot be processed by embedder")
        
        return x


    def res_empty(self, start: int, end: int) -> bool:
        """
        Check if self.res_arr[start:end] is still empty and return True if it is.

        :param start: Starting index of the range to check
        :param end: Ending index of the range to check
        :return: True if the specified range in self.res_arr is empty, False otherwise
        """
        return np.all(self.res_arr[start:end] == 0)

    def embed(self):
        """
        Embed the dataset using the provided model and update the result array with embeddings.

        """
        loader = DataLoader(
            self.dataset, batch_size = self.chunk_size, shuffle = False,
            collate_fn = self.model.preprocess
        )

        is_seq = lambda x : type(x) is list or type(x) is tuple

        # Iterate through the dataset in chunks
        for i, inputs in tqdm(enumerate(loader), total = len(loader)):
            if not self.res_empty(self.progress, self.progress + self.chunk_size):
                self.progress += self.chunk_size
                continue

            if is_seq(inputs):
                inputs = [inp_.to(self.device) for inp_ in inputs]
            else:
                inputs = inputs.to(self.device)

            embeddings = self.model(*inputs if is_seq(inputs) else inputs)
            embeddings = self.postprocess(embeddings)    
        
            # Update the result array with the embeddings
            self.res_arr[self.progress:self.progress + self.chunk_size] = embeddings
            
            # Update progress
            self.progress += self.chunk_size

            if i % self.save_every == 0:
                self.save_state(self.fp)

        # Save when done
        self.save_state(self.fp)
        
    def load_state(self, fp):
        """
        Load the saved state from a file and update the result array and progress

        """
        load_res = joblib.load(fp)

        # Update the result array and progress with the loaded data
        self.res_arr = load_res["embeddings"]
        self.progress = load_res["progress"]

        # Check if the embedding was done
        if load_res["done"]:
            print("Embedding is already done.")


    def save_state(self, fp, compress : int = 0):
        """
        Save the result array and progress so far into a file

        """
        joblib.dump(
            {
                "embeddings" : self.res_arr,
                "progress" : self.progress,
                "done" : (self.progress >= len(self.res_arr))
            },
            fp,
            compress
        )