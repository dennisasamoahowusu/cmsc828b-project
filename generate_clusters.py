import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
os.environ['LASER'] = "../LASER"
sys.path.append("../LASER/source")

import faiss
import numpy as np

# LASER imports
from embed import EmbedLoad
from mine_bitexts import knn


def train_kmeans(fo, d, k, **kwargs):
    """
    Train kmeans on the translations, with k equal to the number
    of prompts
    """
    kmeans = faiss.Kmeans(d, k, **kwargs)
    kmeans.train(fo)
    # Assigments
    dists, clusters = kmeans.index.search(fo, 1)
    dists, clusters = dists.squeeze(), clusters.squeeze()
    return kmeans, dists, clusters


def sample_sentences_from_top_clusters(clusters, n=5, k=None):
    """
    Sample `n` sentences from the `k` largest clusters
    """
    _, cluster_counts = np.unique(clusters, return_counts=True)
    top_clusters = cluster_counts.argsort()[::-1][:k]
    sent_ids = {}
    for cluster_idx in top_clusters:
        sents_in_cluster = np.where(clusters == cluster_idx)[0]
        sampled = np.random.choice(sents_in_cluster, size=n, replace=False)
        sent_ids[cluster_idx] = sampled
    return sent_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing LASER embeddings")

    parser.add_argument("num_clusters", type=int, help="Number of sentence codes")
    parser.add_argument(
        "subtraction_method",
        choices=["none", "mean", "prompt"],
        help=(
            "How to remove semantic information from the LASER representations before"
            "clustering"
        )
    )
    parser.add_argument(
        "--kmeans-iterations", type=int, default=25, help="Number of random k-means starts"
    )
    parser.add_argument(
        "--use_cpu", action="store_true", default=False, help="Do not use GPU"
    )
    parser.add_argument(
        "--normalize", action="store_true", default=False, help="L2-normalize embeddings"
    )

    parser.add_argument(
        "--clusters_to_show",
        type=int, default=10, help="Show examples from this many clusers"
    )
    parser.add_argument(
        "--samples_per_cluster",
        type=int, default=5, help="Show this many sampled sentences from each cluster"
    )    

    parser.add_argument("--seed", type=int, default=11235, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    
    # Read in the embeddings
    input_dir = Path(args.input_dir)
    en_fpath = Path(input_dir, "train.prompts.npy")
    fo_fpath = Path(input_dir, "train.translations.npy")

    en = EmbedLoad(en_fpath)
    fo = EmbedLoad(fo_fpath)
    if args.normalize:
        faiss.normalize_L2(en)
        faiss.normalize_L2(fo) # an inplace operation, used in faiss docs

    # Create the map from prompts to translations
    with open(Path(args.input_dir, "map.json")) as infile:
        sent_map = json.load(infile)

    prompt_to_fo_idx = defaultdict(list)
    for mapping in sent_map:
        prompt_to_fo_idx[mapping['prompt']].append(mapping['trans'])

    # For each prompt, subtract either the mean translation or the english representation
    if args.subtraction_method != "none":        
        fo_subtracted = np.zeros_like(fo)
        for en_idx, fo_indices in prompt_to_fo_idx.items():
            fo_subset = fo[fo_indices]

            if args.subtraction_method == "mean":
                fo_subtracted[fo_indices] = fo_subset - fo_subset.mean(0, keepdims=True)
            if args.subtraction_method == "prompt":
                fo_subtracted[fo_indices] = fo_subset - en[en_idx, None]
    else:
        fo_subtracted = fo # do nothing 

    # Train k-means on the data
    kmeans, dists, clusters = train_kmeans(
        fo_subtracted,
        fo_subtracted.shape[1],
        k=args.num_clusters,
        niter=args.kmeans_iterations,
        gpu=not args.use_cpu,
        seed=args.seed,
    )
    
    # Save cluster assignments
    for mapping, cluster in zip(sent_map, clusters):
        mapping['code'] = int(cluster)

    with open(Path(input_dir, f"train.k-{args.num_clusters}.subtract-{args.subtraction_method}.map.json"), "w") as outfile:
        json.dump(sent_map, outfile)

    # Save centroids
    np.save(Path(input_dir, f"train.k-{args.num_clusters}.subtract-{args.subtraction_method}.centroids.npy"), kmeans.centroids)

    # Print off examples
    if args.clusters_to_show:
        
        en_text_fpath = Path(input_dir, "train.prompts")
        fo_text_fpath = Path(input_dir, "train.translations")
        with open(en_text_fpath, "r") as infile:
            en_text = [l.strip() for l in infile]
        with open(fo_text_fpath, "r") as infile:
            fo_text = [l.strip() for l in infile]

        samples = sample_sentences_from_top_clusters(
            clusters, n=args.samples_per_cluster, k=args.clusters_to_show
        )
        print(
            f"Printing off {args.samples_per_cluster} sentences "
            f"from {args.clusters_to_show} largest clusters"
        )
        for cluster_idx, sent_ids in samples.items():
            print(f"\n\tCluster {cluster_idx:3}")
            for id in sent_ids:
                print(f"{en_text[sent_map[id]['prompt']]:40} | {fo_text[id]}")
        
