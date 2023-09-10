import json

conf = json.load(open("conf.json"))
documents = json.load(open('documents.json'))

nb_of_embeddings = sum([doc['nb_of_embeddings'] for doc in documents])

with open("metadata.json", "w") as f:
    json.dump({"nb_of_documents": len(documents),
               "nb_of_embeddings": nb_of_embeddings },
              f, indent=2)
