# CMI Sensor Behavior Recognition

**Multimodal gesture recognition with CNN ensembles and collection-protocol-aware sequential assignment.**

A personal, approximately one-month post-competition study by **Angelis Steve Nguele**, using wrist-worn inertial, thermal and time-of-flight measurements from [CMI — Detect Behavior with Sensor Data](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data).

## Results

The final submission achieved a **0.864288 private leaderboard score**. Compared with the official leaderboard shown at evaluation, this score falls between the **11th (0.864332)** and **12th (0.862901)** entries, within the gold-medal score range.

This is a post-competition score comparison, not an official competition rank or medal.

| System | Public leaderboard | Private leaderboard |
|---|---:|---:|
| Original hybrid — submission V2 | 0.883769 | 0.862221 |
| Final arithmetic blend — submission V3 | **0.881144** | **0.864288** |

| OOF evaluation, mixed sensor availability | Original hybrid | Final arithmetic blend |
|---|---:|---:|
| Sequence-ID order within each participant | 0.874678 | **0.878749** |
| Mean across five fixed random arrival orders | 0.877123 | **0.880060** |

OOF means out-of-fold: each sequence is predicted by models trained without its validation participants. Arrival-order experiments use the same sequences; they test order sensitivity rather than provide five independent datasets.

## Problem and data

The training set contains **8,151 sequences from 81 participants**, with **18 gesture labels**.

The sensor modalities are:

- **IMU:** acceleration and orientation measurements.
- **THM:** five thermal sensor channels.
- **ToF:** five spatial grids, each containing **8 × 8 distance pixels**, recorded over time.

The task distinguishes eight target behaviors and ten non-target gestures. The evaluation combines binary target/non-target F1 with macro F1 over the eight target classes and a collapsed non-target class.

Our evaluation takes the predicted 18-class gesture first, then collapses non-target gestures for the nine-class metric.

## Approach

### 1. Sensor preprocessing and CNN representations

Three original model variants — **spatial**, **phases** and **heading** — form an equal-weight ensemble.

The pipeline includes IMU feature construction, fold-specific preprocessing, handedness handling, modality availability masks and temporal sequence preparation. The CNN systems use temporal convolutions; ToF grids are processed with spatial encoders.

The phases variant predicts three temporal phases from IMU and uses phase-weighted temporal averaging. The architectures also include residual and squeeze-and-excitation blocks.

The final system uses **no GRU or Transformer**. GRU experiments were evaluated but were not retained.

### 2. Composite prediction targets

A separate CNN predicts **102 situations**, combining:

1. Initial behavior: moves, or relaxes and moves.
2. Body orientation.
3. Gesture.

The collection protocol contains **51 observed orientation–gesture combinations**, each with two initial behaviors. Complete participants have 102 distinct situations. Some participants have incomplete coverage.

### 3. Combining gesture and situation probabilities

Let:

- `p(g)` be the original three-CNN ensemble probability of gesture `g`.
- `r(s)` be the composite CNN probability of situation `s`.
- `m(g)` be the composite probabilities summed over situations belonging to gesture `g`.

The original hybrid preserves the ensemble's gesture probabilities and uses the composite model's conditional distribution within each gesture.

The final blend uses:

```text
gesture_probability(g) = 0.75 * p(g) + 0.25 * m(g)

situation_probability(s)
    = gesture_probability(gesture(s)) * r(s | gesture(s))
```

Here `r(s | g)` is obtained by normalizing the composite model's probabilities within gesture `g`. Zero-mass groups use a uniform conditional fallback.

### 4. Sequential assignment per participant

Predictions are processed as sequences arrive. For each participant, the observed prefix is assigned to distinct situations by minimizing summed negative log-probabilities with the Hungarian algorithm.

Only the currently arriving sequence's gesture is returned. Earlier internal assignments may change when solving a later prefix, but previously emitted predictions are not revised. No future sequences are used.

## Validation

- Five participant-disjoint folds.
- Separate IMU-only, all-sensor and fixed mixed-availability conditions.
- A fixed mixed-availability mask for comparable experiments.
- Sequence-ID order and five random arrival orders using seeds **11, 22, 33, 44, 55**.
- Sequential prefix evaluation rather than treating a full-participant assignment as deployable online inference.
- Checkpoint and preprocessing reproduction checks before submission.

Final arithmetic-blend OOF means across the five random orders:

| Sensor condition | Score |
|---|---:|
| IMU only | 0.853020 |
| All sensors | 0.906139 |
| Fixed mixed availability | 0.880060 |

Weights were compared on these OOF results. These are development comparisons, not a separate nested evaluation of the weight-selection procedure.

## Experiments and findings

- The composite model was weaker than the original ensemble when used independently, but provided useful complementary situation probabilities.
- Arithmetic blending with 25% composite gesture contribution improved the mean score in all three sensor conditions.
- Increasing that contribution to 50% did not improve the mean over the 75/25 blend.
- Giving one original CNN family 50% weight did not improve the tested equal-weight configuration.
- Increasing ToF contrast at inference did not resolve the inspected participant's difficulties and was not retained.
- Sensor audits identified fully invalid orientation measurements in one participant's side-lying recordings.
- Geometry and repetition analyses revealed participant-dependent signal patterns, but did not establish mislabeled examples.

Exploratory analyses using annotated phases or same-participant labeled repetitions are diagnostic studies. They are not presented as submission-time predictors.

## Lessons

1. Reconstruct the experiment and conditions that generated the data.
2. Investigate plausible mechanisms behind observed difficulties.
3. Represent measurements in a way that respects their physical structure.
4. Read the problem statement and understand the role of each data source.
5. Prefer simplicity and justify additional components through measured contributions.
6. Turn concrete hypotheses into experiments, make decisions and stop exhausted lines of investigation.
7. Design validation to match the intended prediction conditions.
8. Evaluate complementarity in the complete system.
9. Preserve code, preprocessing, checkpoints, predictions and experiment provenance.

## Repository status and reproduction

This initial repository documents the final method and verified results.

**The final training and submission source files have not yet been imported into this repository.** Runnable instructions, environment specifications and artifact locations will be added after checking them against the evaluated V3 notebook. This repository is not yet a complete reproduction package.

Competition data and trained weights are not included in this initial commit.

## Acknowledgements

This study benefited from the competition's public discussions and solution write-ups.

In particular, the **second-place solution write-up** supplied the idea of composite 102-situation targets and no-repeat assignment. This project verifies the collection structure, adapts that idea to its CNN pipeline and evaluates hybrid probability construction and sequential blending.

The **first-place solution write-up** also informed investigations of sensor preprocessing and geometry.

Exact source links and code-origin details will be recorded alongside the imported source files.

## Author

**Angelis Steve Nguele**
