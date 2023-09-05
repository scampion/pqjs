import json
import pickle
import sys

import nanopq
import numpy as np


def compute_pq(conf):
    embeddings_file = conf.get('embeddings', 'vectors.bin')
    dim = conf.get('dim', 384)
    M = conf.get('M', 8)
    print("Compute PQ with M ", M)
    X = np.frombuffer(open(embeddings_file, 'rb').read(), dtype="float32")
    X = np.reshape(X, (-1, dim))
    pq = nanopq.PQ(M=M)
    pq.fit(X)
    X_code = pq.encode(X)
    print("vectors ", X_code.shape)
    open("pq.bin", "wb").write(X_code.tobytes())
    with open("codewords.json", "w") as f:
        json.dump(pq.codewords.tolist(), f)
    with open('pq.pkl', 'wb') as f:
        pickle.dump(pq, f)


if __name__ == "__main__":
    conf = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
    compute_pq(conf)
