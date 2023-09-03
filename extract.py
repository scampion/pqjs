import json, os
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

def get_pages_content(filepath):
  for page_number, page_layout in enumerate(extract_pages(open(filepath, 'rb'))):
      yield page_number, [element.get_text() for element in page_layout if isinstance(element, LTTextContainer)]


def pdf2json(filepath):
  for pn, texts in get_pages_content(filepath):
    print(json.dumps({"filename": os.path.basename(filepath), "page_number": pn, "texts": texts}))


if __name__ == "__main__":
  import sys
  filename = sys.argv[1]
  pdf2json(filename)

