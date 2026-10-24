# Changelog

All notable changes to PyRAG Core are recorded here using a lightweight
semantic versioning approach.

## [1.2.0] - 2026-10-03

This release combines the recent `feature/chunk-retrieval-tools` and
`feat/ingestion-parser-strategies` merges into `main`.

### Added

- Metadata filtering and expanded search options for chunk retrieval.
- Retrieval caching and reranker warmup to improve query performance.
- Stronger document and search API coverage across retrieval and ingestion
  flows.

### Changed

- Ingestion now supports multiple parser strategies, including an
  `Unstructured` path and an upgraded PDF parser based on `PyMuPDF4LLM`.
- Document ingestion and pipeline handling now pass parser strategy metadata
  through the API, worker, and UI layers.
- MinIO, Qdrant, and Celery integration paths received resilience updates and
  cleanup.

### Fixed

- Search and retrieval flows now handle metadata-aware chunk selection more
  consistently.
- API and unit tests were updated to match the new retrieval and ingestion
  behavior.

## [1.1.0] - 2026-07-17

This release is built from the current `main` history and captures the work
merged from the feature branches that followed the initial platform baseline.

### Added

- Streamlit UI with integration tests and local setup guidance from
  `enhancement/streamlit-ui-overhaul`.
- Analytics summary endpoint with tests from `fix/analytics-summary-404`.
- Expanded document upload and ingestion handling with improved file
  processing and indexing support from
  `feat/storage-performance-improvements`.

### Changed

- Qdrant search fallback now uses `query_points` when the primary search path is
  not sufficient, improving retrieval resilience from
  `fix/docker-upload-ingestion-loop`.
- The release version is now centralized in `app/core/version.py` so the API,
  package metadata, and app imports stay aligned.
- Local documentation now reflects the current release version.

### Notes

- The current history does not include git tags, so this is the first documented
  release entry.
- Earlier commits included the initial project scaffold, README cleanup, core
  tests, and the first merged fix branch.

## [1.0.0] - 2026-07-17

- Initial project baseline with the core FastAPI application, ingestion
  pipeline, retrieval services, data models, tests, and deployment docs.