# midl

Python client for the [MIDL solar wind dataset](https://csem.engin.umich.edu/MIDL/).

## Install

```
pip install csem-midl
```

## Quickstart

```python
import midl

ds = midl.load("2005-01-01 00:00", "2005-01-01 01:00", "32re")
print(ds)
```

```
<xarray.Dataset> Size: 4kB
Dimensions:  (time: 61)
Coordinates:
  * time     (time) datetime64[ns] 488B 2005-01-01 ... 2005-01-01T01:00:00
Data variables:
    Bx       (time) float64 488B -6.05 -3.94 -3.7 -3.75 ... -2.01 -1.78 -2.08
    By       (time) float64 488B 3.84 3.5 3.12 2.94 3.2 ... 4.97 5.3 5.29 5.2
    Bz       (time) float64 488B -0.93 -4.06 -4.83 -4.92 ... 1.25 2.04 2.05 2.11
    Ux       (time) float64 488B -422.3 -429.9 -433.6 ... -437.4 -441.8 -441.8
    Uy       (time) float64 488B -13.38 -12.91 -9.05 ... -28.15 -27.06 -26.68
    Uz       (time) float64 488B 26.93 47.59 54.86 54.99 ... 0.46 1.42 0.92
    rho      (time) float64 488B 5.169 5.439 5.481 5.359 ... 6.131 6.363 6.323
    T        (time) float64 488B 1.33e+05 1.321e+05 ... 1.159e+05 1.159e+05
Attributes:
    source:             MIDL
    url:                https://csem.engin.umich.edu/MIDL/
    target:             32Re
    midl_propagation:   {'method': 'ballistic', 'target_re': 32.0}
```

The `14re` and `32re` downloads are ballistically propagated server-side.
If you want a different boundary for ballistic propagation, download `l1`
and propagate it yourself:

```python
l1 = midl.load("2024-05-10", "2024-05-11", "l1")

ds_20 = midl.propagate(l1, "ballistic", 20)   # or any target in Re
# equivalent:
ds_20 = l1.midl.propagate("ballistic", 20)
```

`propagate()` reads the per-timestamp source position from the L1 dataset's
`X` column. A `ValueError` is raised if you pass a dataset without `X`
(e.g. a pre-propagated `14re`/`32re` download). If the dataset is already
tagged as propagated, a `UserWarning` is emitted and the call proceeds.

1D MHD-propagated data is also available, pre-computed on the server.
Request it with `target="mhd"` and an integer `target_re` in `[-20, 180]`:

```python
ds_mhd = midl.load("2024-05-10", "2024-05-11", "mhd", target_re=32)
```

Client-side MHD propagation is not supported.

Save to file:

```python
midl.to_csv(ds, "storm.csv")
midl.to_dat(ds, "storm.dat")
```

Data is cached locally after the first download.

## See also

- [MIDL-Pipeline](https://github.com/connordimarco/MIDL-Pipeline) — the code that produces the MIDL dataset
