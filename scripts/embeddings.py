import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
import requests
from joblib import Memory
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from sentence_transformers import SentenceTransformer

from text_splitter import RecursiveCharacterTextSplitter

memory = Memory("cache", verbose=0)

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.FileHandler('embeddings.log'))

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0,
                                          separators=["\n\n", "\n", " ", ""],
                                          keep_separator=False)

@memory.cache
def get_binary_pdf_from_url(url):
    r = requests.get(url)
    return io.BytesIO(r.content)


@memory.cache
def encode(texts, model):
    return model.encode(texts)


@memory.cache
def get_pages_content(url):
    def gen_text(url):
        iofile = get_binary_pdf_from_url(url)
        try:
            for page_number, page_layout in enumerate(extract_pages(iofile)):
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        yield element.get_text()
        except Exception as e:
            print(f"💥 Error with url {url} : " + str(e))

    text = "\n\n".join(gen_text(url))
    return text.replace('. ', '.\n\n')


def compute_embeddings(conf, model):
    documents_file = conf.get('documents', 'documents.json')
    documents = json.load(open(documents_file))
    documents_with_embeddings = []
    with open("embeddings.bin", "wb") as out:
        with ProcessPoolExecutor(max_workers=2) as executor:
            for doc, embeddings in tqdm(executor.map(_embeddings, documents, [model]*len(documents)), total=len(documents), colour='green'):
                documents_with_embeddings.append(doc)
                out.write(embeddings.tobytes())

    with open("documents_with_embeddings.json", "w") as f:
        json.dump(documents_with_embeddings, f, indent=2)


def _embeddings(doc, model):
    global  splitter
    text = get_pages_content(doc['url'])
    texts = [t.replace('\n', '') for t in splitter.split_text(text)]
    embeddings = encode(texts, model)
    doc['nb_of_embeddings'] = embeddings.shape[0]

    log.debug("Embeddings shape " + str(embeddings.shape))
    log.debug("------------------\n".join([f"{len(t)} \n {t} \n" for t in texts]))

    return doc, embeddings


if __name__ == "__main__":
    conf = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
    model_name = conf.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
    model = SentenceTransformer(model_name)
    compute_embeddings(conf, model)
