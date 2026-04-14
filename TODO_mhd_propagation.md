# TODO: Add MHD propagation support

## Context

`midl_pipeline` now writes per-Re 1D-BATSRUS MHD-propagated solar wind as CSV files alongside the existing ballistic outputs. File layout on the web server:

```
data/YYYY/MM/
├── YYYYMM_L1.csv             (unpropagated, existing)
├── YYYYMM_14Re.csv           (ballistic, existing)
├── YYYYMM_32Re.csv           (ballistic, existing)
└── mhd/
    ├── YYYYMM_mhd_000Re.csv
    ├── YYYYMM_mhd_014Re.csv
    ├── YYYYMM_mhd_015Re.csv
    ├── ...
    └── YYYYMM_mhd_190Re.csv   (178 files total per month)
```

Covered Re values: `0` and every integer from `14` to `190` inclusive. Each CSV has the same 8-column schema as the ballistic files: `timestamp, Bx, By, Bz, Ux, Uy, Uz, rho, T`.

## What's currently stubbed

[midl/_propagate.py:111-114](midl/_propagate.py) raises `NotImplementedError` for `method='mhd'`. The public `propagate()` signature already exists and is callable — consumers just can't use it yet.

## What needs to happen

### 1. Extend `TARGETS` fetch logic in [midl/_cache.py](midl/_cache.py)

Current `TARGETS` dict hardcodes `{"14re": "14Re", "32re": "32Re", "l1": "L1"}` → `{base}/{year}/{month}/{year}{month}_{target}.csv`. For MHD we need a different URL shape:

```
{base}/{year}/{month}/mhd/{year}{month}_mhd_{RRR}Re.csv
```

Add an `mhd_url()` or extend `csv_url()` to take a `method` argument that selects the subdirectory and filename pattern. Keep the existing ballistic path untouched.

Valid Re values for MHD: `0, 14, 15, ..., 190`. Reject other integers with a clear error (not silent fallback to nearest), since the server will 404 anyway.

### 2. Fill in the `propagate()` stub in [midl/_propagate.py:111](midl/_propagate.py)

```python
if method == "mhd":
    target_re_int = int(round(target_re))
    if target_re_int != 0 and not (14 <= target_re_int <= 190):
        raise ValueError(
            f"MHD propagation is only available at Re = 0 or 14..190, got {target_re}")
    # Fetch the pre-computed per-Re MHD CSV directly.
    return _load_mhd_csv(ds.start, ds.end, target_re_int)
```

There's no client-side MHD solve — the server already did it. This function just routes the fetch to the right URL.

### 3. Decide on the public API shape

Option A: keep `propagate(ds, method='mhd', target_re=N)` but have it bypass `ds` entirely and do a fresh fetch from the server. Slightly weird — the `ds` argument becomes vestigial for MHD.

Option B: add a parallel `load(start, end, target='mhd_32re')` pattern that lets users fetch MHD directly without going through `propagate()`. Cleaner.

Option C: both — `load()` for direct fetch, `propagate()` for the unified API.

Recommend C: it preserves the existing `propagate()` UX and adds a more natural direct-fetch path.

### 4. Update docs / examples

Show the new usage in the README and docstrings. Make clear that MHD is a pre-computed lookup, not an on-the-fly solve — users requesting non-integer Re will get an error, not interpolation.

### 5. Tests

- Unit: URL builder produces the right path for each valid Re.
- Unit: invalid Re raises clearly.
- Integration (requires server or mock): fetch `mhd_032Re` for a known month and assert the DataFrame shape + column set.
- Parity: MHD at Re=32 should be close to (but not identical to) ballistic at Re=32 on a quiet day; differ significantly through shocks. Useful sanity test, not a strict assertion.

## Not in scope

- Client-side MHD solve — the whole point of the per-Re CSVs is that JS/Python clients don't need an MHD dependency.
- Arbitrary non-integer Re — intentionally rejected. If a user needs 17.3 Re, they either round to 17 or interpolate between 17 and 18 themselves.
- Extending the Re range beyond 0, 14..190 — would require a pipeline-side reprocess.
