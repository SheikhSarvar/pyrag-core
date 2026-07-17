# Releases and Versioning

PyRAG Core uses a simple semantic versioning workflow:

- `MAJOR` for breaking API or schema changes.
- `MINOR` for backward-compatible feature additions.
- `PATCH` for backward-compatible fixes.

## Source of truth

- `app/core/version.py` holds the runtime version exposed by the API and the
  package import.
- `pyproject.toml` mirrors the published package version.
- `CHANGELOG.md` records the human-readable release notes.

## How to release

1. Decide the next version number from the change type.
2. Update `app/core/version.py`.
3. Update `pyproject.toml`.
4. Add a new top section to `CHANGELOG.md`.
5. Update any docs that show the API version.
6. Run the test suite.
7. Tag the commit in git, for example `v1.1.1`.

## Current release

The current documented release is `1.1.0`, which reflects the feature work
merged into `main` after the initial project baseline.
