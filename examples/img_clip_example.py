from LatentCluster.models.clip_model import CLIPEncoder
from LatentCluster.embedder import BaseEmbedder
from LatentCluster.reduction import UMAPReducer
from LatentCluster.utils import clear_tailing_zeros

from datasets import load_dataset

import joblib
import numpy as np
import os
from PIL import Image
import random

path = "openai/clip-vit-base-patch32"


dataset = []
# Assumes theres a dataset of images. You can change this path to whatever you want
data_root = "dataset_images"
sample_size = 20000 
random.seed(0)
for img_path in random.sample(os.listdir(data_root), sample_size):
    im = Image.open(os.path.join(data_root, img_path))
    dataset.append(im)

# Do embedding if it has not already been done

try:
    embs : np.ndarray = joblib.load("embed.joblib")["embeddings"]
    # Assert that the sum along the 512 dimension for each vector in embs is 0
    #for vector in embs:
    #    assert sum(vector) == 0, "The sum of the vector elements is not 0"
except:
    print("Embedding from scratch")
    encoder = CLIPEncoder("IMAGE", path)
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
from LatentCluster.visualization.pygame_vis import PointVis
from LatentCluster.utils import random_sample

print("Running visualization")
#embs, dataset = random_sample((embs, dataset), 5000) # If the  visualization is too slow 
game = PointVis(embs, dataset, width = 1900,height = 1024, mode = "IMAGE")
