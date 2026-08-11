# RTX 4090 pure-forward versus GTP thermal-duration control

Date: 2026-08-06 UTC

## Question

Determine whether the roughly 3300 real nnEval/s GTP result versus roughly
3400 nnEval/s pure-forward result is caused by the longer GTP run heating the
GPU and lowering its boost clock.

## Protocol

- Integration revision: `c4b92e3`
- Exact 19x19 B13, S2, FP16 NHWC; GTP uses t96 and CUDA Graph event pipeline
- Pure-forward increased from 100 to 800 timed iterations, producing a
  6.8-6.9 second timed interval versus 5.8 seconds for ten-position GTP
- ABBA order: pure, GTP, GTP, pure
- 100 ms telemetry for GPU temperature, power, graphics/SM clock, utilization,
  power-cap state, and software/hardware thermal slowdown reasons
- All GPU use serialized by `gpu-lock with --gpu 0`

For GTP, the aligned backend metric is `nnBatches/s * 13`, not the printed
real `nnEval/s`. Every physical launch is padded to B13, whereas the printed
real count excludes dummy rows. All GTP numbers below use this aligned metric.

## Default boost result

| workload | run A aligned nnEval/s | run B aligned nnEval/s | mean aligned nnEval/s |
| --- | ---: | ---: | ---: |
| pure-forward | 3432.09 | 3413.87 | 3422.98 |
| GTP (`nnBatches/s * 13`) | 3348.15 | 3345.16 | 3346.66 |

The matched-duration gap is 76.32 nnEval/s, or 2.28% relative to GTP. The
second pure-forward run was deliberately last and hottest: its timed telemetry
was 70-73 C versus 66-71 C for the first run. Throughput changed by only
-18.23 nnEval/s (-0.53%). No software or hardware thermal slowdown sample was
active. Both workloads were power limited under the default boost policy;
pure-forward actually had a lower mean clock than GTP, so boost behavior does
not explain GTP being slower.

The original unmatched measurements were 3475.31 pure-forward versus
`258.37 * 13 = 3358.81` aligned GTP, a 116.50 nnEval/s gap. Matching the timed
duration reduces that to 76.32 nnEval/s, so 40.18 nnEval/s of the original
difference disappears with duration/order/run-to-run control. That 40.18 is a
combined test-condition effect, not a pure thermal estimate; the directly
observed cool-to-hot pure-forward change was only 18.23 nnEval/s.

## Equal-clock result

A requested 2400 MHz control was not strictly constant because pure-forward
reached the 450 W power limit and ran at 2362-2374 MHz. A second ABBA control
therefore requested 2200 MHz; the hardware selected and held exactly 2205 MHz
for every active sample of every run, with no power-cap or thermal slowdown
sample.

| workload | run A aligned nnEval/s | run B aligned nnEval/s | mean aligned nnEval/s |
| --- | ---: | ---: | ---: |
| pure-forward at 2205 MHz | 3261.27 | 3258.55 | 3259.91 |
| GTP at 2205 MHz (`nnBatches/s * 13`) | 3150.55 | 3144.83 | 3147.69 |

The equal-clock gap is 112.22 nnEval/s, or 3.56% relative to GTP. The two
pure-forward runs differ by only 2.72 nnEval/s (0.083%) despite the second run
being substantially hotter. The gap therefore persists, and becomes larger,
when boost and thermal-frequency variation are removed.

## Conclusion

Test duration and GPU thermal drift are not the primary cause of the
pure-forward to GTP gap. Matching duration removes about 40 nnEval/s from the
original 116.50 nnEval/s difference, while the directly observed thermal
component is about 18 nnEval/s. A 76 nnEval/s aligned gap remains at matched
duration and a 112 nnEval/s gap remains at strict equal clock. The remaining
difference is workload-path overhead: pure-forward measures steady backend
forward latency, while GTP includes request production, graph/event
submission, padding/admission decisions, transfers, output publication,
position ramps/drains, and occasions when search does not have the next batch
ready.

Artifacts:

- `summary.txt`
- `telemetry.csv` and `markers.log`
- `telemetry-fixed2200.csv` and `markers-fixed2200.log`
- `run-controlled-abba.sh` and `run-fixed2200-abba.sh`
- per-run `pure-*.log` and `gtp-*.log`
