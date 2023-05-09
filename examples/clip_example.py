from LatentCluster.models.clip_model import CLIPEncoder
from LatentCluster.embedder import BaseEmbedder
from LatentCluster.reduction import UMAPReducer
from LatentCluster.utils import clear_tailing_zeros

from datasets import load_dataset

import joblib
import numpy as np

path = "openai/clip-vit-base-patch32"


dataset = load_dataset("tweet_eval", "emoji")["test"]["text"] 

# Do embedding if it has not already been done

try:
    embs : np.ndarray = joblib.load("embed.joblib")["embeddings"]
    # Assert that the sum along the 512 dimension for each vector in embs is 0
    #for vector in embs:
    #    assert sum(vector) == 0, "The sum of the vector elements is not 0"
except:
    print("Embedding from scratch")
    encoder = CLIPEncoder("TEXT", path)
    embedder = BaseEmbedder(
        encoder, dataset, 128, "fp16", fp = "embed.joblib"
    )
    embedder.embed()
    embs = joblib.load("embed.joblib")["embeddings"]

embs = clear_tailing_zeros(embs)

# Do reduction if it has not already been done
try:
    reducer = UMAPReducer().load("reducer.joblib")
    embs = reducer(embs)
except:
    print("Fitting UMAP from scratch")
    reducer = UMAPReducer()
    embs = reducer.fit(embs)
    reducer.save("reducer.joblib")

# Do visualization
from LatentCluster.visualization.pygame_vis import PointTextVis
from LatentCluster.utils import random_sample

print("Running visualization")
embs, dataset = random_sample((embs, dataset), 5000)
game = PointTextVis(embs, dataset, width = 1900,height = 1024)
