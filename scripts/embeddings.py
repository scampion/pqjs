import io
import json
import sys

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
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    yield element.get_text()
    except Exception as e:
        print(f"💥 Error with url {url} : " + str(e))


def compute_embeddings(conf, model):
    documents_file = conf.get('documents', 'documents.json')
    documents = json.load(open(documents_file))
    documents_with_embeddings = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0,
                                              separators=["\n\n", "\n", ". ", " ", ""],
                                              keep_separator=False)
    with open("embeddings.bin", "wb") as emb:
        for doc in tqdm(documents):
            text = "\n".join(get_pages_content(doc['url']))
            texts = [t.replace('\n', ' ') for t in splitter.split_text(text)]
            for t in texts:
                log.debug(f"Text length: {len(t)} - ******{t}******")
            embeddings = encode(texts, model)
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
