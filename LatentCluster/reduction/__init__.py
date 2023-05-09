import umap
import joblib
import numpy as np

class UMAPReducer:
    def __init__(self, n_neighbours = 30, min_dist = 0.0, n_components = 2, metric = 'cosine', random_state = 42, low_memory = False):
        self.tform = umap.UMAP(
            n_neighbors=n_neighbours,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=random_state,
            low_memory=low_memory
        )

        self.is_fit = False
        self.n_components = n_components
    
    def fit(self, x : np.ndarray) -> np.ndarray:
        """
        Fit model to some dataset of vectors. Returns the dim reduced vectors.
        """
        return self.tform.fit_transform(x)
    
    def __call__(self, x : np.ndarray) -> np.ndarray:
        """
        Applies reducer to some vectors and returns the result.
        """
        if not self.is_fit:
            print("Warning: Trying to use a reducer that has not been fit to any data")
        return self.tform.transform(x)

    def load(self, fp):
        """
        Load reducer from file
        """
        self.tform = joblib.load(fp)
        self.is_fit = True
        return self
    
    def save(self, fp, compress : int = 0):
        """
        Save reducer to a file
        """
        joblib.dump(
            self.tform, fp, compress
        )