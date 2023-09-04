all: documents.json embeddings.bin pq.bin public

documents.json:
	@echo "Generating documents.json"
	@python3 scripts/documents.py 10000 > documents.json


embeddings.bin: documents.json
	if [ -e embeddings.lock ]; then \
		echo "embeddings.lock exists, stopping here a process is already running"; \
		exit 1; \
	fi
	@echo "Generating embeddings.bin"
	@touch embeddings.lock
	@python3 scripts/embeddings.py
	@rm embeddings.lock


pq.bin: embeddings.bin
	@echo "Generating pq.bin"
	@python3 scripts/pq.py

public:
	@echo "Generating public files"
	@mkdir -p public
	@cp documents.json public/documents.json
	@cp embeddings.bin public/embeddings.bin
	@cp indices.bin public/indices.bin
	@cp codewords.json public/codewords.json
	@cp pq.bin public/pq.bin
	@cp pq.js public/pq.js
	@cp index.html public/index.html


clean:
	@echo "Cleaning up"
	@rm -f embeddings.bin pq.bin indices.bin public/documents.json public/embeddings.bin public/indices.bin public/codewords.json public/pq.bin public/pq.js
