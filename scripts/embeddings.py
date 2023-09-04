import io
import json
import sys

from tqdm import tqdm
import requests
from joblib import Memory
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from sentence_transformers import SentenceTransformer

memory = Memory("cache", verbose=0)

@memory.cache
def get_binary_pdf_from_url(url):
    r = requests.get(url)
    return io.BytesIO(r.content)


def get_pages_content(url):
    iofile = get_binary_pdf_from_url(url)
    try:
      for page_number, page_layout in enumerate(extract_pages(iofile)):
          yield page_number, [element.get_text() for element in page_layout if isinstance(element, LTTextContainer)]
    except Exception as e:
      print(f"💥 Error with url {url} : " + str(e))

def compute_embeddings(conf):
    documents_file = conf.get('documents', 'documents.json')
    documents = json.load(open(documents_file))

    model_name = conf.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
    model = SentenceTransformer(model_name)

    ids = [doc['id'] for doc in documents]
    with open("embeddings.bin", "wb") as emb, open("indices.bin", "wb") as ind:
        for doc in tqdm(documents):
            for _, texts in get_pages_content(doc['url']):
                for embedding in model.encode(texts):
                    emb.write(embedding.tobytes())
                    ind.write(ids.index(doc['id']).to_bytes(8, byteorder='big', signed=False))


if __name__ == "__main__":
    conf = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
    compute_embeddings(conf)



