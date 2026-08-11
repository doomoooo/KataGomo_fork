# RTX 5080 official CUDA backend baseline

This record measures KataGo's official CUDA backend on an NVIDIA GeForce RTX
5080. The SM120 and SM89 optimized backends were explicitly disabled. The
binary reports KataGo v1.17.2, CUDA backend, and CUDA 13.2.86; its SHA-256 is
`92383cdd95ce9a78e83f39527ac43a8de389f0c5bc6e691440add0d276755aca`.
The NVIDIA driver was 595.84.

The scan used the bundled 70M model
(`1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`),
exact 19x19, FP16/NHWC, and two inference streams. Every B4-B32 shape was
measured twice with 200 timed iterations and 80 warmups. Values below are the
two-sample physical nnEval/s medians, rounded to one decimal for display.

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

B9 ranked first and was then confirmed with two independent 1000-iteration
runs after 80 warmups: 1634.6 and 1627.5 physical nnEval/s, median 1631.1 and
relative spread 0.43%. The occupancy wrapper recorded no foreign PID with
nonzero SM activity in any scan or confirmation sample.

The benchmark override fixed `nnMaxBatchSize` to the row's batch,
`numNNServerThreadsPerModel=2`, and disabled `cudaSm120Backend`,
`cudaSm89Backend`, and `cudaSm89Forward`. The full precision config SHA-256 was
`23bae0f1b5315ccdc3d7c05c4f65c54789a10f5811b91cf02a38704a938d30eb`.
