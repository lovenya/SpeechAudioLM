# TTAS G-Eval Audio v6 clean-v2 summary

Rows are retained independently for A2S and S2A. Original split membership and row order are preserved.

| Direction | Split | Source | Kept | Excluded | Kept % |
|---|---:|---:|---:|---:|---:|
| A2S | train | 18,226 | 17,596 | 630 | 96.5434% |
| A2S | val | 2,277 | 2,206 | 71 | 96.8819% |
| A2S | test | 2,277 | 2,197 | 80 | 96.4866% |
| **A2S** | **total** | **22,780** | **21,999** | **781** | **96.5716%** |
| S2A | train | 18,226 | 15,033 | 3,193 | 82.4811% |
| S2A | val | 2,277 | 1,887 | 390 | 82.8722% |
| S2A | test | 2,277 | 1,878 | 399 | 82.4769% |
| **S2A** | **total** | **22,780** | **18,798** | **3,982** | **82.5198%** |

Exclusion is the union of: null final score, final score <= 3, scenario plausibility <= 2, or target-modality success = 1.
