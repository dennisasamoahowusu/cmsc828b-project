
import argparse
import json
import os
import sys
from pathlib import Path
os.environ['LASER'] = "../LASER"
sys.path.append("../LASER/source")

import faiss
import numpy as np

# LASER imports
from embed import EmbedLoad
from mine_bitexts import knn

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--embeddings_fpath", required=True)
    parser.add_argument("--centroids_fpath", required=True)
    parser.add_argument("--output_fpath", required=True)

    args = parser.parse_args()
    
    test = EmbedLoad(args.embeddings_fpath)
    centroids = np.load(args.centroids_fpath)

    # Assign clusters
    sim, indices = knn(test, centroids, 1, use_gpu=False)
    indices = indices.squeeze()
    print(
        f"Assigned {test.shape[0]} sentences to "
        f"{len(np.unique(indices))} unique clusters "
        f"(of a possible {centroids.shape[0]}) "
        f"w/ mean similarity {sim.mean():0.3f}"
    )

    # store the assignments
    sent_assignments = [
        {'trans': i, 'code': int(idx)} for i, idx in enumerate(indices)
    ]
    with open(args.output_fpath, "w") as outfile:
        json.dump(sent_assignments, outfile)
