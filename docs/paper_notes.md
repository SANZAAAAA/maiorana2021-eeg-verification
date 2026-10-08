# Paper notes — Maiorana (2021)

E. Maiorana, *Learning deep features for task-independent EEG-based biometric
verification*, Pattern Recognition Letters **143** (2021) 122–129.
`https://doi.org/10.1016/j.patrec.2021.01.004`

## 1. Goal

Verify a subject's identity from EEG **regardless of the mental task** being
performed, using deep features learned with **channel-specific siamese CNNs**.

## 2. Processing pipeline (Section 3.1)

| Step | Setting |
| --- | --- |
| 1. Band-pass filter | `[8, 30] Hz` (alpha–beta sub-band) |
| 2. Downsampling | `S = 64 Hz` |
| 3. Spatial filter | Common Average Reference (CAR) |
| 4. Framing | `H = 5 s` frames, **80 % overlap** → `1 × 320` per channel |

Input of one frame = `C` sequences of `1 × S·H = 1 × 320` samples, one per
electrode. No artefact removal (tested, no measurable benefit).

## 3. Channel-specific siamese CNN (Table 1)

One network per electrode; the two sub-networks share weights. Conv layers
include batch normalisation.

| # | Layer | Filter | Pad | Input | Output |
| --- | --- | --- | --- | --- | --- |
| L1 | Conv | (1×5×1)×16 | 2 | 1×320×1 | 1×320×16 |
| L2 | ReLU | – | – | 1×320×16 | 1×320×16 |
| L3 | MaxPool | 1×3 | – | 1×320×16 | 1×106×16 |
| L4 | Conv | (1×5×16)×32 | 2 | 1×106×16 | 1×106×32 |
| L5 | ReLU | – | – | 1×106×32 | 1×106×32 |
| L6 | MaxPool | 1×3 | – | 1×106×32 | 1×35×32 |
| L7 | Conv | (1×3×32)×64 | – | 1×35×32 | 1×33×64 |
| L8 | ReLU | – | – | 1×33×64 | 1×33×64 |
| L9 | MaxPool | 1×3 | – | 1×33×64 | 1×11×64 |
| L10 | Conv | (1×3×64)×128 | – | 1×11×64 | 1×9×128 |
| L11 | ReLU | – | – | 1×9×128 | 1×9×128 |
| L12 | MaxPool | 1×3 | – | 1×9×128 | 1×3×128 |
| L13 | Dropout | – | – | 1×3×128 | 1×3×128 |
| L14 | Conv | (1×3×128)×256 | – | 1×3×128 | 1×1×256 |

The output is a **256-dimensional embedding** per frame and per channel.

> The paper writes the padding of the 5-tap convolutions as `[0,2]`; the
> input/output dimensions listed in Table 1 (320→320 and 106→106) are only
> consistent with a total padding of 4, i.e. **2 on each side**. The
> implementation uses `padding=2`.

## 4. Contrastive loss (Eq. 1)

```
L = (1 - y) · ½ · D²  +  y · ½ · max(0, d - D)²
```

`D` = Euclidean distance between the two embeddings, `y = 0` for a genuine
pair (same subject), `y = 1` for an impostor pair, `d` = margin.

## 5. Training (Section 4.1)

* genuine pairs: frames of the same subject from **different sessions**
  (and, for the multi-protocol model, possibly different tasks);
* for each genuine pair, **2 impostor pairs** with another subject's frame;
* **SGDM**, batch size **128**, learning rate **1e-3**, weight decay **5e-3**;
* first three sessions + fifth for training, fourth for validation;
* MatConvNet on an Nvidia GPU.

## 6. Verification (Section 3.1)

* a **one-class SVM** per subject and per channel, fitted on the enrolment
  frame representations;
* `C` per-channel scores per probe → **score fusion** → single decision.

## 7. Experiments (Section 4)

* 45 subjects, 5 sessions, 6 protocols, 19 electrodes (proprietary database);
* subject-disjoint 5-fold cross-validation (30 train / 15 test);
* **STD** (< 1 month) vs **LTD** (~15 months);
* **SSE** (single-session enrolment) vs **MSE** (multiple-session enrolment);
* Table 3 = within-task, Table 4 = task-independent, Table 5 = per-channel.

## 8. Substitutions in this reproduction

The paper's database is **not public**, so the pipeline is run on the
**PhysioNet EEGMMIDB** substitute (see `README.md` for the full list of
deviations).
