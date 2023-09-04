all: documents.json embeddings.bin pq.bin public

documents.json:
	@echo "Generating documents.json"
	@python3 scripts/collect_documents.py 10000 > documents.json


embeddings.bin: documents.json
	@echo "Generating embeddings.bin"
	@python3 scripts/embeddings.py


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
	@rm -f documents.json embeddings.bin pq.bin indices.bin public/documents.json public/embeddings.bin public/indices.bin public/codewords.json public/pq.bin public/pq.js
