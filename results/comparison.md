# Deep (channel-specific siamese CNN) vs hand-crafted (AR+MFCC)

_Configuration_: 10 subjects, 19 channels, 5 folds, 10 epochs, 40 pairs/subject.

## A. Within-task (paper Table 3 analogue)

| Scenario | Hand-crafted (EER %) | Deep (EER %) |
|---|---|---|
| T1|MSE_STD | 7.5 | 18.7 |
| T1|SSE_STD | 5.3 | 17.0 |
| T2|MSE_STD | 10.3 | 17.1 |
| T2|SSE_STD | 5.9 | 20.7 |

## B. Task-independent (paper Table 4 analogue)

| Cross-task verification | Hand-crafted (EER %) | Deep, single-proto training (EER %) | Deep, multi-proto training (EER %) |
|---|---|---|---|
| single-task enrolment, `SSE_STD` | 5.5 | 16.4 | 17.2 |
| single-task enrolment, `MSE_STD` | 8.3 | 17.5 | 16.3 |

## C. Per-channel capability (paper Table 5 analogue)

| Channel | Hand-crafted (EER %) | Deep (EER %) |
|---|---|---|
| F8 | 5.4 | 7.3 |
| F4 | 14.1 | 14.0 |
| C4 | 13.6 | 17.1 |
| F3 | 8.2 | 18.2 |
| F1 | 11.5 | 23.3 |
| P8 | 4.3 | 23.4 |
| P7 | 12.4 | 23.6 |
| F2 | 14.5 | 23.8 |
| C3 | 12.0 | 23.8 |
| Fz | 9.4 | 24.7 |
| Cz | 12.9 | 28.5 |
| T8 | 10.7 | 29.8 |
| P4 | 14.9 | 30.3 |
| T7 | 17.2 | 31.5 |
| Pz | 16.5 | 31.6 |
| P3 | 14.5 | 31.6 |
| F7 | 14.9 | 33.8 |
| O2 | 15.1 | 35.8 |
| O1 | 14.9 | 38.5 |

_(Deep numbers are per-fold means; see `tables.md` for the pooled estimates and standard deviations.)_
