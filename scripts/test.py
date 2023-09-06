import json
import sys
import numpy as np
import pickle
import math

from sentence_transformers import SentenceTransformer

def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return left



conf = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
model_name = conf.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
model = SentenceTransformer(model_name)

#query = "Audiovisual rights in sports events"
#query = "Protection of journalists and human rights defenders from manifestly unfounded or abusive court proceedings"
#query = "analyses the kinds of compensation available to victims of climate change disasters in the EU"
#query = " ".join(sys.argv[2:])
#query = "organic pet food"

query = open("query.txt").read()
query = model.encode(query)


pq = pickle.load(open("pq.pkl", "rb"))
query_q = pq.encode(vecs=np.array([query]))[0]

X_code = np.frombuffer(open("pq.bin", 'rb').read(), dtype="uint8")
X_code = np.reshape(X_code, (-1, conf["M"]))
print("vectors ", X_code.shape)
dists = pq.dtable(query=query).adist(codes=X_code)

documents = json.load(open("documents_with_embeddings.json"))
indices = [doc['nb_of_embeddings'] for doc in documents]
for i in range(1, len(indices)):
    indices[i] += indices[i - 1]

max_features = 20
results = {}

print(np.sort(dists)[:max_features])

print(np.min(dists)/np.sort(dists)[:max_features])

for i in np.argsort(dists)[:max_features]:
    doc_i = binary_search(indices, i)
    results[doc_i] = results.get(doc_i, 0) + np.min(dists)/dists[i]

k_max = 100
print("PQ sort")
pq_r = np.argsort(dists)[:k_max]
print(pq_r) if k_max < 20 else None

X = np.frombuffer(open("vectors.bin", 'rb').read(), dtype="float32")
X = np.reshape(X, (-1, 384))
dists_exact = np.linalg.norm(X - query, axis=1) ** 2
print("exact sort")
ex_r = np.argsort(dists_exact)[:k_max]
print(ex_r) if k_max < 20 else None

print("Intersection ratio ", len(set(pq_r).intersection(set(ex_r))) / k_max)
print("-"*80)

#results = {k: len(v)/(sum(v)/len(v)) for k, v in results.items()}

# normalize
results = {k: v / sum(results.values()) for k, v in results.items()}
#sort by value
results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
#enrich with documents metadata
results = [{"rank": i, "score": v, **documents[k]} for i,  (k, v) in enumerate(results.items())]
print(json.dumps(results, indent=2))
