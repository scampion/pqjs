#
# Copyright (c) 2023 Sebastien Campion.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import json
import os
import sys
import numpy as np
import pickle
import math
from collections import defaultdict

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

documents = json.load(open("documents.json"))
indices = [doc['nb_of_embeddings'] for doc in documents]
for i in range(1, len(indices)):
    indices[i] += indices[i - 1]

conf = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
model_name = conf.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
model = SentenceTransformer(model_name)

query_text = " ".join(sys.argv[2:])
query = model.encode(query_text)
k_max = 100


# TEST FULL SCAN
if os.path.exists("embeddings.bin"):
    print("################ Embeddings results")
    X = np.frombuffer(open("embeddings.bin", 'rb').read(), dtype="float32")
    X = np.reshape(X, (-1, 384))
    dists_exact = np.linalg.norm(X - query, axis=1) ** 2
    print("Exact sort")
    ex_r = np.argsort(dists_exact)[:k_max]
    print(ex_r)


    #########################
    # counter with distance
    doc_counter = defaultdict(list)
    for i in ex_r:
        doc_i = binary_search(indices, i)
        doc_counter[doc_i].append(dists_exact[i])
    #sort by value
    print()
    print("Full scan results counter sorted by value as dists\n---------------------------------")
    doc_counter = {k: v for k, v in sorted(doc_counter.items(), key=lambda item: np.mean(item[1]), reverse=False)}
    for i, (k, v) in enumerate(doc_counter.items()):
        print(f"{np.mean(v):0.4f}", "----", documents[k]['title'])
        if i > 10:
            break

    docs_exact = doc_counter.keys()

##################

pq = pickle.load(open("pq.pkl", "rb"))
pq.verbose = False
query_q = pq.encode(vecs=np.array([query]))[0]
print("query q", query_q[:6])

# print("codewords 0 0 :")
# print(pq.codewords[0].shape)
# ds = [np.linalg.norm(query[24:48] - pq.codewords[0][i], ord=2) for i in range(256)]
# print(ds)
# print(min(ds))
# print(np.argmin(ds))

# print("Query 24")
# print(query[24:48])


print("################ PQ Results")

X_code = np.frombuffer(open("pq.bin", 'rb').read(), dtype="uint8")
X_code = np.reshape(X_code, (-1, conf["M"]))
print("pq codes shape : ", X_code.shape)
dists = pq.dtable(query=query).adist(codes=X_code)

doc_counter = defaultdict(list)

for i in np.argsort(dists)[:k_max]:
    doc_i = binary_search(indices, i)
    doc_counter[doc_i].append(dists[i])

#sort by value
print()
print("PQ code results counter sorted by value as dists\n---------------------------------")
doc_counter = {k: v for k, v in sorted(doc_counter.items(), key=lambda item: np.mean(item[1]), reverse=False)}
for i, (k, v) in enumerate(doc_counter.items()):
    print(f"{np.mean(v):0.4f}", "----", documents[k]['title'])
    if i > 10:
        break

pq_r = np.argsort(dists)[:k_max]
print("Intersection ratio emb", len(set(pq_r).intersection(set(ex_r))) / k_max)
print("Intersection ratio docs", len(set(doc_counter.keys()).intersection(set(docs_exact))) / len(doc_counter))
#     print("-" * 80)


#
#
#
# print("PQ sort")
# pq_r = np.argsort(dists)[:k_max]
# print(pq_r) if k_max < 20 else None
#
# # normalize
# results = {k: v / sum(results.values()) for k, v in results.items()}
# # sort by value
# results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
# # enrich with documents metadata
# results = [{"rank": i, "score": v, **documents[k]} for i, (k, v) in enumerate(results.items())]
# print(json.dumps([r['title'] for r in results][:12], indent=2))
#
# if os.path.exists("vectors.bin"):
#     print("################ Vectors Results")
#     X = np.frombuffer(open("vectors.bin", 'rb').read(), dtype="float32")
#     X = np.reshape(X, (-1, 384))
#     dists_exact = np.linalg.norm(X - query, axis=1) ** 2
#     print("exact sort")
#     ex_r = np.argsort(dists_exact)[:k_max]
#     print(ex_r) if k_max < 20 else None
#
#     print("Intersection ratio ", len(set(pq_r).intersection(set(ex_r))) / k_max)
#     print("-" * 80)

# results = {k: len(v)/(sum(v)/len(v)) for k, v in results.items()}
#    print("Intersection ratio ", len(set(pq_r).intersection(set(ex_r))) / k_max)
#    print("-" * 80)
sys.exit(0)
