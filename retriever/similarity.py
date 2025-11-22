import numpy as np

def cosine_similarity(vec1, vec2):
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    return np.dot(vec1, vec2)

def inner_product(vec1, vec2):
    return np.dot(vec1, vec2)

def l2_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def compute_similarity_matrix(query_vec, doc_vecs, metric="cosine"):
    results = []
    for vec in doc_vecs:
        if metric == "cosine":
            results.append(cosine_similarity(query_vec, vec))
        elif metric == "ip":
            results.append(inner_product(query_vec, vec))
        elif metric == "l2":
            results.append(l2_distance(query_vec, vec))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    return np.array(results)
