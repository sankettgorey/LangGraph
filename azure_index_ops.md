🔹 1. Index Lifecycle Operations

These define and manage the index itself.

✅ Create index

Define fields

Set key field

Configure searchable / filterable / sortable fields

Define vector fields (dimensions, profiles)

Define analyzers

Define semantic configuration

Define scoring profiles

❌ Modify index schema (limited)

You can only:

Add new fields (non-breaking)

Add semantic configurations

Add scoring profiles

Add synonym maps

You cannot:

Change field types

Change analyzers

Change vector dimensions

Remove fields

Change key field

➡ Anything structural → index migration

✅ Delete index

Removes schema + all indexed data

Does NOT delete source data

✅ Get index / list indexes

Inspect schema

Validate field configuration

Used in migrations & debugging

🔹 2. Data Operations (Documents)

These operate on documents inside the index.

✅ Upload documents

Insert new documents

Requires unique key

Bulk upload supported

✅ Merge documents

Update specific fields

Partial update

Does not overwrite entire document

✅ Merge or upload

Upsert behavior

Most commonly used in ingestion pipelines

✅ Delete documents

Delete by key

Delete by filter

Bulk delete supported

❌ Update vector dimensions

Not allowed

Requires migration

🔹 3. Query Operations

These are read operations.

✅ Keyword (lexical) search

BM25-based full-text search

Supports:

searchFields

queryType (simple / full)

filters

facets

sorting

✅ Vector search

kNN / ANN search

HNSW

Supports:

cosine / dot / euclidean

hybrid queries

filtering with vectors

✅ Hybrid search

Keyword + vector combined

Weighted scoring

Most common in GenAI apps

✅ Semantic search

Reranks results using semantic models

Supports:

captions

answers

highlights

✅ Faceted search

Aggregations

Useful for UI filters

✅ Filter-only queries

Structured lookup

No text search

✅ Autocomplete / suggest

Typeahead

Search-as-you-type

Requires suggester configuration

🔹 4. Vector & AI-Specific Operations

These power GenAI / RAG scenarios.

✅ Define vector fields

Collection(Edm.Single)

Dimension-specific

Attached to vector search profile

✅ Configure vector search profiles

HNSW parameters

Metric type

Profile reuse

✅ Change embedding model (via migration)

New dimension

New vector field

New index

✅ Hybrid retrieval for RAG

Vector + keyword + filter

Standard RAG pattern

🔹 5. Semantic & NLP Features
✅ Semantic configuration

Title field

Content fields

Keyword fields

✅ Semantic answers

Extractive QA-style responses

✅ Semantic captions

Highlighted summaries

✅ Synonym maps

Custom synonyms

Expand query recall

🔹 6. Indexer Operations (If using indexers)

Only applies if you use Azure-managed data sources.

✅ Create indexer

Connects data source → index

Can include skillsets

✅ Run indexer

Manual or scheduled

Reindex on demand

✅ Reset indexer

Full reprocessing

Used during migration

✅ Monitor indexer status

Success / failure

Error diagnostics

🔹 7. Skillset Operations (AI Enrichment)

If using built-in AI enrichment.

✅ Create skillsets

OCR

Key phrase extraction

Entity recognition

Text split

Embedding generation (Azure OpenAI)

✅ Attach skillset to indexer

Enrichment pipeline

❌ Modify skillset output schema

Usually requires index migration

🔹 8. Security & Access Operations
✅ API keys management

Admin keys

Query keys

✅ Role-based access (RBAC)

Managed identity

Azure AD integration

✅ Private endpoints

Network isolation

🔹 9. Monitoring & Diagnostics
✅ Query metrics

Latency

Throughput

✅ Index size monitoring

Document count

Storage usage

✅ Logs & diagnostics

Indexer failures

Query failures

🔹 10. Migration & Versioning Operations (Operational Pattern)

These are patterns, not APIs — but critical.

✅ Index versioning

index_v1, index_v2

Blue–green deployment

✅ Reindex data

From source

From stored chunks

Without re-uploading files

✅ Zero-downtime migration

Shadow index

Traffic switch

🔹 11. What Azure AI Search is NOT

❌ Not a primary data store
❌ Not a transactional DB
❌ Not mutable like SQL
❌ Not a vector DB replacement for training

It is a serving & retrieval engine.