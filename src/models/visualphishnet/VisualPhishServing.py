from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from skimage.transform import resize
from tools.ModelServing import ModelServing

from eval_new import find_names_min_distances
from Evaluate import Evaluate
from ModelHelper import ModelHelper


class VisualPhishServing(ModelServing):
    def __init__(self, args=None):
        super().__init__()
        self.args = args

    async def predict(self, data: dict):
        """Implementation of the predict method for VisualPhish"""
        img = data.get("image", None)
        url = data.get("url", None)
        resized = resize(img, (self.args.reshape_size[0], self.args.reshape_size[1]), anti_aliasing=True)
        print(f"Resized image shape: {resized.shape}")
        if len(resized.shape) == 3:
            resized = np.expand_dims(resized, axis=0)
        data_emb = self.model.predict(resized, verbose=0)
        pairwise_distance = Evaluate.compute_all_distances_batched(data_emb, self.targetlist_emb)

        min_distances = None
        only_names = None

        for i in range(data_emb.shape[0]):
            distances_to_target = pairwise_distance[i, :]
            idx, values = Evaluate.find_min_distances(np.ravel(distances_to_target), 1)
            names_min_distance, only_names, min_distances = find_names_min_distances(idx, values, self.all_file_names)

        cls = 1 if float(min_distances) <= self.args.threshold else 0
        target = str(only_names[0]) if cls else "unknown"

        return {
            "url": str(url),
            "class": cls,
            "target": target,
            "distance": float(min_distances),
        }

    async def on_startup(self):
        """Startup logic (e.g., loading resources)"""
        self.targetlist_emb, self.all_labels, self.all_file_names = self.load_targetemb(
            self.args.emb_dir / "whitelist_emb.npy",
            self.args.emb_dir / "whitelist_labels.npy",
            self.args.emb_dir / "whitelist_file_names.npy",
        )

        modelHelper = ModelHelper()
        self.model = modelHelper.load_model(self.args.emb_dir, self.args.saved_model_name, self.args.margin).layers[3]

    def load_targetemb(self, emb_path, label_path, file_name_path):
        targetlist_emb = np.load(emb_path)
        all_labels = np.load(label_path)
        all_file_names = np.load(file_name_path)
        return targetlist_emb, all_labels, all_file_names


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--emb-dir", type=Path, default="/code/model")
    parser.add_argument("--margin", type=float, default=2.2)
    parser.add_argument("--saved-model-name", type=str, default="model2")
    parser.add_argument("--reshape-size", default=[224, 224, 3])
    parser.add_argument("--threshold", type=float, default=8)
    args = parser.parse_args()

    serving = VisualPhishServing(args)
    serving.run()

# in api folder
# uv run python -m visualphishnet.VisualPhishServing
