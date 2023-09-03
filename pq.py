import nanopq
import numpy as np

X = np.frombuffer(open("embeddings.bin", 'rb').read(), dtype="float32")
X = np.reshape(X, (-1, 384))
pq = nanopq.PQ(M=8)
pq.fit(X)
X_code = pq.encode(X)
np.save(open("X_code",'wb'), X_code)
open("X_code_","wb").write(X_code.tobytes())

np.save(open("codewords",'wb'), pq.codewords)

with open('pq.pkl', 'wb') as f:
  pickle.dump(pq, f)

query = "Audiovisual rights in sports events"
from sentence_transformers import SentenceTransformer

sentences = [query]
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embedding = model.encode(sentences)[0]

dists = pq.dtable(query=embedding).adist(codes=X_code)
results = list(np.argsort(dists)[:5])
results.reverse()
for r in results:
    print(open("index.json").readlines()[r])

