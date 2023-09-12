all: documents.json embeddings.bin pq.bin metadata.json public

documents.json:
	@echo "Generating documents.json"
	@python3 scripts/documents.py 10000 > documents.json


embeddings.bin: documents.json
	@if [ -e embeddings.lock ]; then \
		echo "embeddings.lock exists, stopping here a process is already running"; \
		exit 1; \
	fi
	@echo "Generating embeddings.bin"
	@touch embeddings.lock
	@python3 scripts/embeddings.py
	@rm embeddings.lock


pq.bin: embeddings.bin
	@echo "Generating pq.bin"
	@python3 scripts/pq.py conf.json

metadata.json: pq.bin
	@echo "Generating metadata.json"
	@python3 scripts/metadata.py


public:
	@echo "Generating public files"
	@mkdir -p public
	@cp documents_with_embeddings.json public/documents.json
	@cp vectors.bin public/vectors.bin
	@cp codewords.json public/codewords.json
	@cp pq.bin public/pq.bin
	@cp conf.json public/conf.json


clean:
	@echo "Cleaning up"
	@rm -f documents_with_embeddings.json embeddings.bin embeddings.lock pq.pkl vectors.bin pq.bin codewords.json embeddings.log public/documents.json public/vectors.bin public/codewords.json public/pq.bin public/pq.js
