# Phase 5 — Hybrid Retrieval & Reranking

## Goal
Add multi-stage search pipeline: broad FAISS retrieval → SQLite metadata filter → hybrid scoring → optional cross-encoder reranking. Replace in-memory JSON with SQLite for queryable object/track storage.

## Files Modified
- `backend/pipeline.py` — add `MetadataDB` class, integrate into `index_video()`
- `backend/search_engine.py` — add `Reranker` class, multi-stage pipeline, SQLite filter
- `backend/config.py` — reranker settings + granular hybrid weights
- `backend/.env.example` — document new settings
- `backend/requirements.txt` — add `sentence-transformers`

## Acceptance Criteria
- [ ] SQLite DB created per video with objects + tracks tables
- [ ] Indexing populates SQLite alongside existing JSON
- [ ] Search supports optional SQLite metadata filter (by class)
- [ ] Cross-encoder reranker works as optional post-processing stage
- [ ] All existing 44 tests pass
- [ ] No new files created (changes in existing files only)
- [ ] Backward compatible: old indexes without SQLite fall back gracefully
