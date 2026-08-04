# PR 153 pressure benchmark summary

| Run | Mode | Duration (s) | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | P99 TPOT (ms) | D2H GB/s | D2H p99 (ms) | H2D GB/s | H2D p99 (ms) | Peak NPU MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | native | 74.06 | 5818.00 | 20563.50 | 113.35 | 143.93 | 9.51 | 78.84 | 4.20 | 121.66 | 38382 |
| 2 | mapped | 75.27 | 6180.62 | 20948.92 | 114.10 | 147.48 | 10.25 | 75.12 | 36.27 | 8.72 | 38422 |
| 3 | mapped | 74.23 | 6112.86 | 20857.57 | 112.06 | 144.92 | 7.88 | 70.72 | 38.49 | 8.64 | 38422 |
| 4 | native | 76.02 | 6053.60 | 20768.59 | 113.86 | 145.75 | 8.13 | 96.47 | 3.27 | 177.98 | 38378 |
| 5 | native | 75.16 | 6067.69 | 21145.12 | 115.18 | 147.04 | 7.96 | 97.38 | 2.83 | 205.79 | 38382 |
| 6 | mapped | 67.82 | 4377.67 | 17342.14 | 100.08 | 126.35 | 7.99 | 66.90 | 20.72 | 27.47 | 38418 |

## Three-lifecycle means

| Mode | Duration (s) | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | P99 TPOT (ms) | D2H GB/s | D2H p99 (ms) | H2D GB/s | H2D p99 (ms) | Peak NPU MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native | 75.08 | 5979.76 | 20825.74 | 114.13 | 145.57 | 8.48 | 96.47 | 3.34 | 205.79 | 38381 |
| mapped | 72.44 | 5557.05 | 19716.21 | 108.75 | 139.59 | 8.58 | 70.72 | 29.23 | 27.47 | 38421 |

## Mapped versus native

- Duration: -3.52%
- Mean TTFT: -7.07%
- P99 TTFT: -5.33%
- Mean TPOT: -4.72%
- P99 TPOT: -4.11%
- D2H aggregate bandwidth: +1.17%
- D2H p99 latency: -26.69%
- H2D aggregate bandwidth: +774.73%
- H2D p99 latency: -86.65%
- Peak NPU process memory: +0.10%
