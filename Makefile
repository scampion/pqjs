all: documents.json embeddings.bin

documents.json:
	@echo "Generating documents.json"
	@python3 scripts/collect_documents.py 10000 > documents.json


embeddings.bin: documents.json
	@echo "Generating embeddings.bin"
	@python3 scripts/embeddings.py

