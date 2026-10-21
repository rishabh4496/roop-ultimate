
**Love** (900 frames, reference = the second dense run; the two dense runs are bit-identical)

| K | pre-pass | detector calls / frame | faces found | frames with a face | frames matched to a source | flow-placed error median / p95 (% eye dist.) | linear interpolation median | placed without flow: n, median |
|---|---|---|---|---|---|---|---|---|
| 1 | 23.8 s (+0%) | 1.00 | 1667 (+0%) | 826 / 826 | 621 (+0%) | - | - | - |
| 2 | 21.1 s (-11%) | 0.54 | 1583 (-5%) | 819 / 826 | 603 (-3%) | 1.22 / 11.1 | 3.94 | 193, 25.7 |
| 4 | 18.3 s (-23%) | 0.31 | 1497 (-10%) | 802 / 826 | 571 (-8%) | 1.15 / 9.2 | 6.51 | 315, 28.9 |
| 8 | 17.0 s (-29%) | 0.19 | 1341 (-20%) | 755 / 826 | 431 (-31%) | 1.28 / 9.8 | 12.67 | 306, 39.5 |

**MonicaBellucci** (900 frames, reference = the second dense run; the two dense runs are bit-identical)

| K | pre-pass | detector calls / frame | faces found | frames with a face | frames matched to a source | flow-placed error median / p95 (% eye dist.) | linear interpolation median | placed without flow: n, median |
|---|---|---|---|---|---|---|---|---|
| 1 | 24.4 s (+0%) | 1.00 | 2306 (+0%) | 900 / 900 | 872 (+0%) | - | - | - |
| 2 | 21.0 s (-14%) | 0.51 | 2258 (-2%) | 896 / 900 | 870 (-0%) | 1.07 / 11.8 | 3.34 | 223, 27.6 |
| 4 | 18.2 s (-26%) | 0.26 | 2149 (-7%) | 887 / 900 | 864 (-1%) | 1.20 / 14.1 | 4.48 | 379, 19.7 |
| 8 | 17.0 s (-30%) | 0.14 | 1980 (-14%) | 873 / 900 | 770 (-12%) | 1.41 / 11.7 | 6.36 | 427, 19.4 |

**Weeds** (900 frames, reference = the second dense run; the two dense runs are bit-identical)

| K | pre-pass | detector calls / frame | faces found | frames with a face | frames matched to a source | flow-placed error median / p95 (% eye dist.) | linear interpolation median | placed without flow: n, median |
|---|---|---|---|---|---|---|---|---|
| 1 | 17.1 s (+0%) | 1.00 | 1221 (+0%) | 900 / 900 | 719 (+0%) | - | - | - |
| 2 | 15.1 s (-12%) | 0.51 | 1225 (+0%) | 888 / 900 | 680 (-5%) | 0.80 / 13.1 | 1.22 | 88, 20.1 |
| 4 | 13.6 s (-21%) | 0.26 | 1187 (-3%) | 880 / 900 | 541 (-25%) | 0.89 / 14.8 | 1.80 | 125, 14.6 |
| 8 | 11.7 s (-31%) | 0.13 | 1115 (-9%) | 868 / 900 | 718 (-0%) | 1.14 / 17.8 | 3.64 | 165, 13.1 |

**Point-tracking back ends** (ms per call, median of the probe repetitions; NVOF: this OpenCV build has no CUDA device (the opencv-python wheels ship without CUDA))

| crop px | points | cv (CPU) | torch (CPU) | torch (CUDA) |
|---|---|---|---|---|
| 96 | 40 | 0.14 | 15.9 | 17.4 |
| 192 | 70 | 0.23 | 26.5 | 20.3 |
| 192 | 300 | 0.30 | 83.4 | 22.1 |
| 384 | 1000 | 0.66 | 140.3 | 21.3 |
| 768 | 3000 | 2.34 | 295.4 | 21.5 |
