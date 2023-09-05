import json
import sys
import numpy as np
import pickle


from sentence_transformers import SentenceTransformer

conf = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
model_name = conf.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
model = SentenceTransformer(model_name)

query = "Audiovisual rights in sports events"
query = "Protection of journalists and human rights defenders from manifestly unfounded or abusive court proceedings"
query = "analyses the kinds of compensation available to victims of climate change disasters in the EU"

emb = model.encode(query)


pq = pickle.load(open("pq.pkl", "rb"))
e = pq.encode(vecs=np.array([emb]))

X_code = np.frombuffer(open("pq.bin", 'rb').read(), dtype="uint8")
X_code = np.reshape(X_code, (-1, 8))
print("vectors ", X_code.shape)
dists = pq.dtable(query=emb).adist(codes=X_code)

documents = json.load(open("documents_with_embeddings.json"))
indices = [doc['nb_of_embeddings'] for doc in documents]
for i in range(1, len(indices)):
    indices[i] += indices[i - 1]


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


results = {}
for i in np.argsort(dists):
    doc_i = binary_search(indices, i)
    results[doc_i] = results.get(doc_i, 0) + 1
    if len(results) > 5:
        break

# normalize
results = {k: v / sum(results.values()) for k, v in results.items()}
#sort by value
results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
#enrich with documents metadata
results = [{"rank": i, "score": v, **documents[k]} for i,  (k, v) in enumerate(results.items())]
print(json.dumps(results, indent=2))
