import json
from sentence_transformers import SentenceTransformer

sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#embeddings = model.encode(sentences)
#print(embeddings)


if __name__ == "__main__":
  import sys
  with open("embeddings.bin", "ab") as bin, open("index.json", "a") as ind:
    for l in open(sys.argv[1],'r').readlines():
      data = json.loads(l)
      for embedding in model.encode(data["texts"].copy()):
        bin.write(embedding.tobytes())
        if "texts" in data.keys():
          data.pop("texts")
        ind.write(json.dumps(data)+"\n")
    


