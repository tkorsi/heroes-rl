# Synthetic analysis fixture

Hand-written input for `tests/test_analysis.py`. It mimics the shape of a schema-v2
`random-baseline` run directory but was **not produced by VCMI** and is **not game
evidence**. Engine, map and policy hashes are placeholders; only the `policy.jsonl`
hash is real so that ledger validation can be exercised. Tests copy this directory
before deriving variants and never modify it in place.
