# RTX 5080 official backend baselines

This record measures the standard CUDA and TensorRT backend paths on an NVIDIA
GeForce RTX 5080. All custom SM89/SM120 backends and TensorRT FlashAttention
plugins were disabled. Both paths used the same 70M model
(`1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`),
exact 19x19, FP16/NHWC, and two inference streams. The NVIDIA driver was
595.84. Values are physical nnEval/s and are rounded to one decimal for display.

## CUDA

The CUDA binary reports KataGo v1.17.2 and CUDA 13.2.86; its SHA-256 is
`92383cdd95ce9a78e83f39527ac43a8de389f0c5bc6e691440add0d276755aca`.
Every B4-B32 shape was measured twice with 200 timed iterations and 80 warmups.

| Batch | nnEval/s | Batch | nnEval/s | Batch | nnEval/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1380.4 | 14 | 1505.6 | 24 | 1525.2 |
| 5 | 1433.6 | 15 | 1565.3 | 25 | 1519.0 |
| 6 | 1562.6 | 16 | 1537.4 | 26 | 1523.6 |
| 7 | 1597.4 | 17 | 1549.1 | 27 | 1485.8 |
| 8 | 1612.8 | 18 | 1564.8 | 28 | 1508.7 |
| 9 | 1620.1 | 19 | 1559.9 | 29 | 1466.2 |
| 10 | 1461.2 | 20 | 1509.3 | 30 | 1478.0 |
| 11 | 1462.1 | 21 | 1516.1 | 31 | 1475.6 |
| 12 | 1484.0 | 22 | 1524.7 | 32 | 1463.5 |
| 13 | 1518.1 | 23 | 1518.8 | | |

B9 ranked first and was confirmed with two independent 1000-iteration runs
after 80 warmups: 1634.6 and 1627.5 nnEval/s, median 1631.1 and relative spread
0.43%.

## TensorRT

The non-plugin TensorRT binary reports KataGo v1.17.1 and TensorRT 10.16.1.11
on CUDA 13.2; its SHA-256 is
`d6b82958232cd58affea188db9516be61bcdf715fa0220f0c9ca0afd902ef86f`.
Existing engine caches were used only to avoid rebuild time. B4-B32 were each
measured with three one-second repetitions after 20 warmups.

| Batch | nnEval/s | Batch | nnEval/s | Batch | nnEval/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1570.3 | 14 | 1942.7 | 24 | 1802.9 |
| 5 | 1639.7 | 15 | 1932.7 | 25 | 1793.6 |
| 6 | 1811.1 | 16 | 1955.7 | 26 | 1792.9 |
| 7 | 1831.5 | 17 | 1972.0 | 27 | 1747.6 |
| 8 | 1888.0 | 18 | 1971.3 | 28 | 1745.0 |
| 9 | 1936.4 | 19 | 1922.8 | 29 | 1750.1 |
| 10 | 1887.6 | 20 | 1855.8 | 30 | 1742.2 |
| 11 | 1945.2 | 21 | 1863.8 | 31 | 1732.5 |
| 12 | 1971.8 | 22 | 1841.2 | 32 | 1768.3 |
| 13 | 1951.5 | 23 | 1805.7 | | |

B17 ranked first and was confirmed with five two-second repetitions after 20
warmups: 2032.6, 2029.7, 2026.7, 2026.1, and 2022.6 nnEval/s; the median was
2026.7.

The occupancy wrapper covered every scan and confirmation. It recorded no
foreign PID with nonzero SM activity. Both benchmark paths excluded
preprocessing, postprocessing, H2D, D2H, and search from device-only forward
timing.
