# Release checklist

## PyPI project

The import package is `cmg`, but the PyPI distribution name is
`claim-memory-graph` because `cmg` is already used by another PyPI project.

Configure PyPI Trusted Publishing for:

- owner: `MatteoLeonesi`
- repository: `claim-memory-graph-sdk`
- workflow: `publish.yml`
- environment: `pypi`

## Local verification

Use a clean output directory so stale ignored artifacts from older package names
cannot be mixed into a manual check or upload.

```bash
DIST_DIR="$(mktemp -d)"
uv build --out-dir "$DIST_DIR"
uv run --with twine python -m twine check "$DIST_DIR"/*
uv run --with "$DIST_DIR"/claim_memory_graph-0.1.1-py3-none-any.whl python -c "import cmg; print(cmg.__version__)"
```

## Publish

Create and publish a GitHub release for the version tag. The `Publish` workflow
builds the source distribution and wheel, then uploads them to PyPI using
trusted publishing.
