# RTX 4090 D official backend baselines

This record measures the standard CUDA and TensorRT backend paths on an NVIDIA
GeForce RTX 4090 D. All custom SM89/SM120 backends were disabled. Both paths
used the same 70M model
(`1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`),
exact 19x19, FP16/NHWC, and two inference streams. The NVIDIA driver was
590.48.01. Values are physical nnEval/s and are rounded to one decimal for
display.

## CUDA

The CUDA binary reports KataGo v1.17.2 and CUDA 13.2.86; its SHA-256 is
`c45aa608c4d332bf878a48c4319ded3ed77f90a53f8984427c4d953204491c12`.
Every B4-B32 shape was measured twice with 200 timed iterations and 80 warmups.

| Batch | nnEval/s | Batch | nnEval/s | Batch | nnEval/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1561.3 | 14 | 1849.5 | 24 | 1729.6 |
| 5 | 1657.2 | 15 | 1876.7 | 25 | 1715.5 |
| 6 | 1752.5 | 16 | 1840.4 | 26 | 1686.0 |
| 7 | 1620.0 | 17 | 1821.4 | 27 | 1658.1 |
| 8 | 1702.6 | 18 | 1813.4 | 28 | 1648.2 |
| 9 | 1743.2 | 19 | 1820.4 | 29 | 1614.3 |
| 10 | 1833.4 | 20 | 1796.7 | 30 | 1601.6 |
| 11 | 1841.8 | 21 | 1791.4 | 31 | 1568.3 |
| 12 | 1874.0 | 22 | 1778.9 | 32 | 1563.0 |
| 13 | 1883.8 | 23 | 1744.3 | | |

B13 ranked first and was confirmed with two independent 1000-iteration runs
after 80 warmups: 1894.6 and 1885.0 nnEval/s, median 1889.8 and relative spread
0.51%.

## TensorRT

The standard TensorRT binary reports KataGo v1.17.1 and TensorRT 10.16.1.11 on
CUDA 13.1; its SHA-256 is
`04fe09bf1587b20d457b6accfb4b44821e1ba52e96384906d49ca0bb815dfdb7`.
Existing engine caches were used where available; missing exact-batch engines
were built before their timed measurements. B4-B32 were each measured with
three one-second repetitions after 20 warmups.

| Batch | nnEval/s | Batch | nnEval/s | Batch | nnEval/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1748.8 | 14 | 2015.0 | 24 | 1730.4 |
| 5 | 1859.8 | 15 | 2009.9 | 25 | 1680.1 |
| 6 | 1957.3 | 16 | 2012.2 | 26 | 1657.7 |
| 7 | 1924.1 | 17 | 2027.5 | 27 | 1611.2 |
| 8 | 2019.1 | 18 | 2018.1 | 28 | 1578.3 |
| 9 | 2007.2 | 19 | 1999.8 | 29 | 1569.2 |
| 10 | 2040.3 | 20 | 1946.0 | 30 | 1563.3 |
| 11 | 2071.2 | 21 | 1860.8 | 31 | 1517.7 |
| 12 | 2101.1 | 22 | 1794.8 | 32 | 1558.2 |
| 13 | 2077.3 | 23 | 1735.4 | | |

B12 ranked first and was confirmed with five two-second repetitions after 20
warmups: 2354.0, 2345.5, 2339.7, 2332.4, and 2326.8 nnEval/s; the median was
2339.7.

The occupancy wrapper covered every scan and confirmation. It recorded no
foreign PID with nonzero SM activity. Both benchmark paths excluded
preprocessing, postprocessing, H2D, D2H, and search from device-only forward
timing.
