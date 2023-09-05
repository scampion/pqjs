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


@memory.cache
def encode(texts, model):
    return model.encode(texts)


def get_pages_content(url):
    iofile = get_binary_pdf_from_url(url)
    try:
        for page_number, page_layout in enumerate(extract_pages(iofile)):
            yield page_number, [element.get_text() for element in page_layout if isinstance(element, LTTextContainer)]
    except Exception as e:
        print(f"💥 Error with url {url} : " + str(e))


def compute_embeddings(conf, model):
    documents_file = conf.get('documents', 'documents.json')
    documents = json.load(open(documents_file))
    documents_with_embeddings = []

    with open("embeddings.bin", "wb") as emb:
        for doc in tqdm(documents):
            all_texts = [text for _, texts in get_pages_content(doc['url']) for text in texts]
            embeddings = encode(all_texts, model)
            emb.write(embeddings.tobytes())
            doc['nb_of_embeddings'] = len(embeddings)
            documents_with_embeddings.append(doc)

    with open("documents_with_embeddings.json", "w") as f:
        json.dump(documents_with_embeddings, f, indent=2)


if __name__ == "__main__":
    conf = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
    model_name = conf.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
    model = SentenceTransformer(model_name)
    compute_embeddings(conf, model)
