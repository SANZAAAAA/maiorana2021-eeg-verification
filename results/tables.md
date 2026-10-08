# Reproduction results (EER, %)

- subjects: 10, channels: 19, folds: 5


## A. Within-task (Table 3 analogue)

| Protocol / Scenario | EER % (mean ± std over folds) | EER % (pooled) |
|---|---|---|
| T1|MSE_STD | 8.2 ± 10.4 | 18.7 |
| T1|SSE_STD | 10.7 ± 13.2 | 17.0 |
| T2|MSE_STD | 15.4 ± 17.2 | 17.1 |
| T2|SSE_STD | 11.3 ± 15.0 | 20.7 |

## B. Task-independent (Table 4 analogue)

| Training / Verification | EER % (mean ± std over folds) | EER % (pooled) |
|---|---|---|
| train_multi_protocol|single_enrol_cross_verify|MSE_STD | 9.4 ± 11.4 | 16.3 |
| train_multi_protocol|single_enrol_cross_verify|SSE_STD | 7.9 ± 11.2 | 17.2 |
| train_single_protocol|single_enrol_cross_verify|MSE_STD | 9.2 ± 10.5 | 17.5 |
| train_single_protocol|single_enrol_cross_verify|SSE_STD | 8.8 ± 10.9 | 16.4 |

## C. Per-channel capability (Table 5 analogue)

| Channel | EER % |
|---|---|
| F8 | 7.3 ± 14.6 |
| F4 | 14.0 ± 21.2 |
| C4 | 17.1 ± 17.9 |
| F3 | 18.2 ± 20.1 |
| F1 | 23.3 ± 21.1 |
| P8 | 23.4 ± 20.9 |
| P7 | 23.6 ± 19.3 |
| F2 | 23.8 ± 20.1 |
| C3 | 23.8 ± 18.2 |
| Fz | 24.7 ± 19.1 |
| Cz | 28.5 ± 21.8 |
| T8 | 29.8 ± 20.2 |
| P4 | 30.3 ± 22.7 |
| T7 | 31.5 ± 20.7 |
| Pz | 31.6 ± 22.1 |
| P3 | 31.6 ± 18.6 |
| F7 | 33.8 ± 18.8 |
| O2 | 35.8 ± 18.5 |
| O1 | 38.5 ± 18.6 |
