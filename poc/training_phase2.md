# Training Phase 2: aligning V-JEPA 2 clips with UCA sentences, and per-sentence zone / crime-type heads

**Goal:** go from "this video contains a crime" (phase 1) to **sentence-level** understanding: for any short clip,
(1) put it in the same space as a natural-language description, (2) say how alarming it is (GREEN / YELLOW / RED),
and (3) say which crime it looks like.

**Headline (official test split, 1968 UCA sentences from 286 videos):**

| | Metric | Phase 2 | Baseline |
|---|---|---|---|
| Clip -> sentence retrieval | R@1 / R@5 / R@10 | **0.024 / 0.075 / 0.127** | chance 0.0005 / 0.0025 / 0.0051 |
| Sentence -> clip retrieval | R@5 | **0.078** | chance 0.0025 |
| Zone, alignment head's zone branch | macro-F1 | **0.502** | 0.462 (linear probe on raw V-JEPA) |
| Zone, ordinal Head A (thresholds 0.5/0.5) | macro-F1 | 0.488 | same linear probe |
| Crime type, 6 groups (GT YELLOW/RED sentences) | macro-F1 | **0.396** | 0.35 (phase 1, video level) |
| Crime type, 13 classes (GT YELLOW/RED sentences) | macro-F1 | **0.269** | 0.23 (phase 1, video level) |

---

## 1. Core ideas

1. **One clip per sentence.** UCA (UCF-Crime Annotation) gives every video time-stamped sentences. Each sentence
   becomes one training pair: the 16 frames inside its [start, end] span <-> its text. That turns 1,854 weakly
   labelled videos into 23,595 precisely localised samples.
2. **Frozen encoders on both sides.** Video: V-JEPA 2 ViT-L, the same encoder as phase 1 and the live demo.
   Text: OpenAI `text-embedding-3-small` (1536-d). Only a small MLP is trained to map the 1024-d clip vector into
   the 1536-d text space, so training takes under a minute.
3. **Contrastive alignment (CLIP-style InfoNCE).** In each batch of 2048 pairs, a clip must score its own sentence
   above the other 2047, and vice versa. Many UCA sentences are near-duplicates ("A man walks into the store"), so
   pairs whose texts have cosine > 0.95 are **not** used as negatives. The temperature is learnable (init 0.07).
4. **Zones as a second task.** GPT labelled each sentence GREEN (normal), YELLOW (suspicious / aftermath) or RED
   (harm happening now), using only past context. Zones are predicted two ways: as a branch of the alignment head,
   and as a standalone ordinal head (A) where output 1 = "at least YELLOW" and output 2 = "is RED".
5. **Crime type only where it matters.** Head B is trained only on YELLOW/RED sentences of crime videos, so it learns
   what the incident looks like, not what the empty parking lot before it looks like.

## 2. Data

| | |
|---|---|
| Frames | Dropbox mp4s -> HDF5, 224x224, 7.5 fps (phase-1 cache), 64 shards + index.csv |
| Sentences | UCA, matched to our video ids: 23,595 sentences, 22,705 unique strings, 1,854 videos |
| Zone labels | GPT (`uca_pipeline.py` stage 1): GREEN 17,936 (76.0%), YELLOW 3,438 (14.6%), RED 2,221 (9.4%), 386 low-confidence |
| Splits | official UCF-Crime train/test (from the HDF5 attrs). Validation = held-out TRAIN videos, never single sentences |
| Alignment head split | train 18986 / val 2641 sentences (10% of train videos, random) / test 1968 |
| Heads A/B split | train 17304 (1332 videos) / val 4323 (236 videos, 15%, stratified by class) / test 1968 |

The test split is crime-heavy (GREEN 1,219 / YELLOW 498 / RED 251, 38% non-green) compared with val (20% non-green),
so val scores overstate test scores.

## 3. What we did

| Step | What | Where | Time |
|---|---|---|---|
| 1. Text embeddings | `align.py text`: every sentence -> `text-embedding-3-small`, deduplicated, batches of 1000, cached per call, L2-normalised | local machine (API key) | 69 s, 23 API calls |
| 2. Clip embeddings | `align.py clips`: per sentence, 16 frames uniformly from [start, end] (span widened to >= 16 frames around its center; frames repeat if the video is shorter), upscaled 224 -> 256 (bicubic, V-JEPA mean/std), V-JEPA 2 ViT-L, mean-pooled to 1024-d | 48 work parts claimed via lock files by 1x H200, 1x H100, 3x V100 at once | 16 min wall clock |
| 3a. Alignment head | `align.py train`: MLP 1024 -> 1024 (LayerNorm, GELU, dropout 0.1) -> 1536 (L2-norm) + zone branch; symmetric InfoNCE + 0.5 x weighted zone CE; batch 2048, AdamW 1e-3, wd 0.05, cosine, early stop on val R@5 | CPU | 32 epochs, 41 s |
| 3b. Zone Head A + crime-type Head B | `zone_heads.py`: 8-config sweep each (lr x dropout x hidden), early stop on val macro-F1, threshold tuning for A | CPU | 246 s total |
| 4. Evaluation | `align.py eval` (retrieval, zone metrics, linear baseline, 40 examples) and `zone_heads.py` tables | CPU | 105 s |

Clip-embedding throughput was limited by HDF5 reads from `/projects` (about 90 MB/s per stream that day), not by the GPUs:

| GPU | Parts | Clips | Clips/s |
|---|---|---|---|
| H100 80GB | 17 | 8,355 | 24.6 |
| H200 NVL | 10 | 4,914 | 11.4 |
| V100-SXM2 32GB (3 jobs) | 21 | 10,326 | 8.5 |

Detours, for the record:
- A first run read the Kaggle 64x64 PNGs. We switched to the 224 px HDF5 frames before training, and that run was cancelled.
- A wandb integration was dropped because `wandb` is not in the `mllm3` env.
- The gpu-partition H200 queue was about 32 h, so the GPU work ran on the `sharing` partition plus V100s, and the small heads ran on CPU.
- A 1968-sentence encoding timing (280 s) in `results/phase2_times.json` comes from the cancelled PNG run. Ignore it.

### Sweeps

Head A (ordinal zone), val macro-F1 at thresholds 0.5/0.5:

| lr | dropout | hidden | val macro-F1 | best epoch |
|---|---|---|---|---|
| 0.001 | 0.1 | 256 | 0.555 | 21 |
| 0.001 | 0.1 | 512 | 0.573 | 39 |
| 0.001 | 0.3 | 256 | 0.571 | 58 |
| 0.001 | 0.3 | 512 | 0.567 | 29 |
| 0.0003 | 0.1 | 256 | 0.524 | 3 |
| 0.0003 | 0.1 | 512 | 0.547 | 35 |
| 0.0003 | 0.3 | 256 | 0.519 | 3 |
| 0.0003 | 0.3 | 512 | 0.550 | 78 |

Head B best configs: 13 classes lr 0.001, dropout 0.3, hidden 256 (val macro-F1 0.356);
6 groups lr 0.001, dropout 0.1, hidden 256 (val macro-F1 0.416).
Head B peaks after 2-11 epochs, so it overfits quickly: there are only a few thousand YELLOW/RED training sentences.

Alignment head, validation curve (selected checkpoint: epoch 22):

| epoch | train loss | tau | val R@1 | val R@5 | val R@10 | val zone acc |
|---|---|---|---|---|---|---|
| 1 | 6.708 | 0.070 | 0.006 | 0.027 | 0.051 | 0.678 |
| 2 | 5.645 | 0.069 | 0.009 | 0.038 | 0.067 | 0.708 |
| 5 | 4.653 | 0.067 | 0.017 | 0.053 | 0.093 | 0.696 |
| 10 | 3.794 | 0.063 | 0.020 | 0.062 | 0.102 | 0.694 |
| 15 | 3.241 | 0.060 | 0.017 | 0.065 | 0.113 | 0.716 |
| 20 | 2.839 | 0.058 | 0.019 | 0.064 | 0.101 | 0.743 |
| 22 | 2.711 | 0.057 | 0.019 | 0.069 | 0.111 | 0.739 |
| 25 | 2.533 | 0.056 | 0.019 | 0.064 | 0.106 | 0.736 |
| 30 | 2.308 | 0.055 | 0.019 | 0.064 | 0.105 | 0.746 |
| 32 | 2.240 | 0.055 | 0.019 | 0.059 | 0.101 | 0.756 |

## 4. What we learned

1. **Retrieval works, but only coarsely.** Clip -> sentence R@10 is 12.7% against 0.5% chance, so
   about 25x better than random. In absolute terms it is low. Val R@5 plateaus at about 0.07 by epoch 12 while the
   train loss keeps falling, so the head overfits the training sentences. The examples below show why: the top-3
   usually describe the *right kind of scene and event* (cars colliding, someone taking goods from a shelf) but from a
   different video. Surveillance scenes and UCA phrasing are both highly repetitive, so the exact sentence is hard to single out.
2. **The text loss helps the zone task a little.** The alignment head's zone branch reaches macro-F1
   0.502 against 0.462 for a linear probe on the same frozen features
   (RED recall 0.62 vs 0.56). It also slightly beats the dedicated ordinal Head A
   (0.488).
3. **The RED-recall >= 0.8 target was not reachable on val.** The best was 0.77, at the lowest grid thresholds
   (t_yellow 0.1, t_red 0.05). The tuner then fell back to maximising RED recall, which floods predictions
   with RED: test macro-F1 0.424 vs 0.488 at 0.5/0.5, but RED->GREEN drops from
   20.7% to 5.2%.
   For an alert system that trade-off may be acceptable; for a dashboard it is not. Choose per use.
4. **YELLOW is the hard class** (test F1 0.26-0.36 for every model). It mixes "suspicious before" and "aftermath",
   which look very different, and its GPT labels are the least consistent.
5. **Per-sentence crime type beats per-video crime type.** 6-group macro-F1 0.396 vs 0.35 in
   phase 1, and 13-class 0.269 vs 0.23. Feeding Head B only the incident sentences removes most of the
   background. In the cascade (Head A decides which sentences Head B sees) the scores drop only slightly
   (0.367 / 0.252).

## 5. Results (official test split)

### 5.1 Retrieval

| Direction | R@1 | R@5 | R@10 |
|---|---|---|---|
| Clip -> sentence (over all 1968 test sentences) | 0.024 | 0.075 | 0.127 |
| Sentence -> clip | 0.023 | 0.078 | 0.118 |
| Chance | 0.0005 | 0.0025 | 0.0051 |

Ties count as hits, so identical sentence texts are interchangeable.

### 5.2 Zone: alignment head's zone branch

| Class | n | Precision | Recall | F1 | Accuracy (one-vs-rest) |
|---|---|---|---|---|---|
| GREEN | 1219 | 0.846 | 0.633 | 0.725 | 0.702 |
| YELLOW | 498 | 0.360 | 0.392 | 0.375 | 0.670 |
| RED | 251 | 0.304 | 0.622 | 0.408 | 0.770 |
| **macro** | 1968 | 0.503 | 0.549 | 0.502 | overall accuracy 0.571 |

| true \ predicted | GREEN | YELLOW | RED |
|---|---|---|---|
| GREEN | 772 | 282 | 165 |
| YELLOW | 110 | 195 | 193 |
| RED | 30 | 65 | 156 |

**RED predicted as GREEN: 12.0%** (30 of 251 RED sentences).

### 5.3 Zone: baseline linear classifier on raw V-JEPA embeddings (no text loss)

| Class | n | Precision | Recall | F1 | Accuracy (one-vs-rest) |
|---|---|---|---|---|---|
| GREEN | 1219 | 0.854 | 0.609 | 0.711 | 0.693 |
| YELLOW | 498 | 0.346 | 0.345 | 0.346 | 0.669 |
| RED | 251 | 0.233 | 0.558 | 0.328 | 0.709 |
| **macro** | 1968 | 0.477 | 0.504 | 0.462 | overall accuracy 0.536 |

| true \ predicted | GREEN | YELLOW | RED |
|---|---|---|---|
| GREEN | 742 | 244 | 233 |
| YELLOW | 97 | 172 | 229 |
| RED | 30 | 81 | 140 |

RED predicted as GREEN: 12.0%.

### 5.4 Zone: ordinal Head A (best sweep config: lr 0.001, dropout 0.1, hidden 512)

Thresholds 0.5 / 0.5:

| Class | n | Precision | Recall | F1 | Accuracy (one-vs-rest) |
|---|---|---|---|---|---|
| GREEN | 1219 | 0.806 | 0.726 | 0.764 | 0.722 |
| YELLOW | 498 | 0.377 | 0.341 | 0.358 | 0.691 |
| RED | 251 | 0.274 | 0.458 | 0.343 | 0.776 |
| **macro** | 1968 | 0.486 | 0.509 | 0.488 | overall accuracy 0.595 |

| true \ predicted | GREEN | YELLOW | RED |
|---|---|---|---|
| GREEN | 885 | 197 | 137 |
| YELLOW | 161 | 170 | 167 |
| RED | 52 | 84 | 115 |

Tuned thresholds t_yellow 0.1 / t_red 0.05 (RED-recall-first, see section 4.3):

| Class | n | Precision | Recall | F1 | Accuracy (one-vs-rest) |
|---|---|---|---|---|---|
| GREEN | 1219 | 0.911 | 0.548 | 0.684 | 0.687 |
| YELLOW | 498 | 0.301 | 0.225 | 0.257 | 0.672 |
| RED | 251 | 0.213 | 0.733 | 0.330 | 0.621 |
| **macro** | 1968 | 0.475 | 0.502 | 0.424 | overall accuracy 0.490 |

| true \ predicted | GREEN | YELLOW | RED |
|---|---|---|---|
| GREEN | 668 | 206 | 345 |
| YELLOW | 52 | 112 | 334 |
| RED | 13 | 54 | 184 |

RED predicted as GREEN: 5.2%.

### 5.5 Crime type: Head B, 6 groups (GT YELLOW/RED sentences of test crime videos)

| Class | n | Precision | Recall | F1 | Accuracy (one-vs-rest) |
|---|---|---|---|---|---|
| Arrest | 37 | 0.171 | 0.324 | 0.224 | 0.885 |
| FireExplosion | 99 | 0.492 | 0.293 | 0.367 | 0.861 |
| RoadAccidents | 63 | 0.455 | 0.889 | 0.602 | 0.897 |
| Theft | 306 | 0.673 | 0.647 | 0.660 | 0.717 |
| Vandalism | 33 | 0.054 | 0.091 | 0.067 | 0.885 |
| Violence | 183 | 0.580 | 0.377 | 0.457 | 0.773 |
| **macro** | 721 | 0.404 | 0.437 | 0.396 | overall accuracy 0.509 |

| true \ predicted | Arrest | FireExplosion | RoadAccidents | Theft | Vandalism | Violence |
|---|---|---|---|---|---|---|
| Arrest | 12 | 3 | 3 | 10 | 0 | 9 |
| FireExplosion | 12 | 29 | 14 | 25 | 5 | 14 |
| RoadAccidents | 1 | 2 | 56 | 1 | 1 | 2 |
| Theft | 20 | 12 | 20 | 198 | 40 | 16 |
| Vandalism | 0 | 1 | 5 | 15 | 3 | 9 |
| Violence | 25 | 12 | 25 | 45 | 7 | 69 |

Cascade (sentences Head A flags YELLOW/RED with the tuned thresholds): accuracy 0.489, macro-F1 0.367.

### 5.6 Crime type: Head B, 13 classes

| Class | n | Precision | Recall | F1 | Accuracy (one-vs-rest) |
|---|---|---|---|---|---|
| Abuse | 7 | 0.000 | 0.000 | 0.000 | 0.954 |
| Arrest | 37 | 0.127 | 0.216 | 0.160 | 0.883 |
| Arson | 39 | 0.310 | 0.333 | 0.321 | 0.924 |
| Assault | 17 | 0.071 | 0.059 | 0.065 | 0.960 |
| Burglary | 127 | 0.530 | 0.488 | 0.508 | 0.834 |
| Explosion | 60 | 0.556 | 0.167 | 0.256 | 0.920 |
| Fighting | 35 | 0.368 | 0.200 | 0.259 | 0.945 |
| RoadAccidents | 63 | 0.557 | 0.698 | 0.620 | 0.925 |
| Robbery | 26 | 0.009 | 0.038 | 0.015 | 0.816 |
| Shooting | 124 | 0.256 | 0.089 | 0.132 | 0.799 |
| Shoplifting | 119 | 0.962 | 0.429 | 0.593 | 0.903 |
| Stealing | 34 | 0.279 | 0.559 | 0.373 | 0.911 |
| Vandalism | 33 | 0.143 | 0.303 | 0.194 | 0.885 |
| **macro** | 721 | 0.321 | 0.275 | 0.269 | overall accuracy 0.329 |

Cascade: accuracy 0.303, macro-F1 0.252.
Small classes (Abuse 7, Assault 17, Robbery 26 test sentences) swing a lot on a handful of sentences.

## 6. Qualitative: retrieval for 20 RED + 20 YELLOW test clips

Each clip's aligned embedding is matched against the text embeddings of **all 1968 GT test sentences**. Shown:
the true sentence, then the top-3 retrieved sentences with their GPT zone and cosine. In these 40 examples, the true
sentence is in the top-3 2 times and in the top-10 7 times. 41 of the 120 retrieved sentences share the
query's zone, and 14 come from the same video. The predicted zone is from the alignment head's zone branch.

**1. Shoplifting022_x264**: true zone RED, predicted RED, true sentence ranked #139 of 1968

- GT: *The boy squatted for a moment, stepped forward and took a box of things from the bottom of the shelf, and hid it in his clothes;*
- top-1 [YELLOW, cos 0.28]: A man and a woman were looking at the products in the middle of the supermarket shelves, and then the man took something from the corner shelf
- top-2 [GREEN, cos 0.28]: A man comes to get something from the shelf
- top-3 [RED, cos 0.27]: The old man stole his wallet

**2. RoadAccidents131_x264**: true zone RED, predicted RED, true sentence ranked #7 of 1968

- GT: *At the intersection, a black car going straight collided with a black car going straight on the other road.*
- top-1 [RED, cos 0.36]: A reverse-driving motorcycle was hit by a black car at a corner
- top-2 [RED, cos 0.29]: A black car collided with a straight black car while turning at a traffic light
- top-3 [RED, cos 0.28]: A white car and a black car collided at the intersection

**3. RoadAccidents010_x264**: true zone RED, predicted RED, true sentence ranked #192 of 1968

- GT: *A red bus driven by a man in white hit a white car parked on the roadside, and also hit a walking bald man in a black coat and white shirt.*
- top-1 [GREEN, cos 0.33]: A blue car drove to the upper right side of the street and stopped
- top-2 [RED, cos 0.32]: A blue car collided with a yellow car
- top-3 [GREEN, cos 0.31]: A white car stopped on the side of the road.

**4. Robbery102_x264**: true zone RED, predicted RED, true sentence ranked #1 of 1968

- GT: *The man in black and the woman in the courtyard were fighting for the bag, and both sides were pulling.*
- top-1 [RED, cos 0.37] **(the GT sentence)**: The man in black and the woman in the courtyard were fighting for the bag, and both sides were pulling.
- top-2 [RED, cos 0.33] (same video): A man in black ran into the yard, holding a knife in his hand, and then the woman in black stood up and stepped aside.
- top-3 [RED, cos 0.31]: The woman reached out and took something and put it in her bag

**5. Vandalism036_x264**: true zone RED, predicted RED, true sentence ranked #133 of 1968

- GT: *Then the shirtless man pushed all the shelves to the ground while walking towards the store.*
- top-1 [YELLOW, cos 0.28]: A man wearing a white shirt and a red hat ran out of the room
- top-2 [YELLOW, cos 0.28]: Then jump out from the counter and leave.
- top-3 [YELLOW, cos 0.28]: The man ran out of the room holding a box.

**6. Burglary035_x264**: true zone RED, predicted YELLOW, true sentence ranked #19 of 1968

- GT: *The man lay on the window and looked into the room. Finally he opened the window and climbed into the room through the window.*
- top-1 [RED, cos 0.41]: A man came to the door and sprinkled gasoline, mainly in front of the door.
- top-2 [RED, cos 0.35]: The man finally pried open the door and entered the house.
- top-3 [RED, cos 0.35]: The man standing in front of the door kicked hard at the door.

**7. Shooting011_x264**: true zone RED, predicted RED, true sentence ranked #304 of 1968

- GT: *Pointing guns at several ladies*
- top-1 [YELLOW, cos 0.34]: The man in white holds a shovel and talks to the people in the yard.
- top-2 [GREEN, cos 0.34]: Man in white shirt akimbo and talking to people around him
- top-3 [GREEN, cos 0.31]: A man in white clothes in the room picked up the phone next to him and held a bright flashlight in his other hand

**8. Shoplifting005_x264**: true zone RED, predicted YELLOW, true sentence ranked #70 of 1968

- GT: *The man in the green down jacket opened the zipper and cooperated with the man in the black down jacket to quickly stuff a black thing into the down jacket.*
- top-1 [YELLOW, cos 0.35]: A man in black clothes and a black hat walked to the door of a store with something in his hand. The door of the store was covered with glass.
- top-2 [GREEN, cos 0.33]: A man in black with short hair stood in the hall with his arms akimbo
- top-3 [YELLOW, cos 0.32] (same video): The man in the black down jacket walked out of the surveillance corner with a black object in his hand

**9. Arson011_x264**: true zone RED, predicted GREEN, true sentence ranked #6 of 1968

- GT: *A man entered the elevator and kept lighting the camera on a corner of the elevator with a lighter.*
- top-1 [GREEN, cos 0.38]: A woman in the elevator was chatting and laughing and pressed the elevator button
- top-2 [RED, cos 0.37]: The yellow-haired woman touched the oil gun, and the oil gun suddenly exploded and caught fire.
- top-3 [YELLOW, cos 0.34]: The yellow-haired woman threw the oil gun on the ground, and there was a flame at the muzzle of the oil gun.

**10. Fighting018_x264**: true zone RED, predicted RED, true sentence ranked #29 of 1968

- GT: *The woman kicked the man in the lower body and made him kneel on the ground.*
- top-1 [RED, cos 0.38]: Several people came and beat each other in the hall again. Two people were knocked down and then pulled up.
- top-2 [RED, cos 0.33]: One person continued to beat the man who fell to the ground and walked inside
- top-3 [RED, cos 0.32]: The man quickly fell to the ground.

**11. Shoplifting004_x264**: true zone RED, predicted RED, true sentence ranked #448 of 1968

- GT: *The hunchbacked old man put the items on the table into his own plastic bag*
- top-1 [GREEN, cos 0.35]: The man in pink took out the money and put it on the counter
- top-2 [RED, cos 0.33]: The man took something from the counter and put it in his pocket
- top-3 [GREEN, cos 0.31]: The man returned to the counter and handed something to the clerk, who gave him something

**12. Assault006_x264**: true zone RED, predicted RED, true sentence ranked #145 of 1968

- GT: *The black boy was pushed down to the bed.*
- top-1 [RED, cos 0.42]: Three men beat a man with sticks until someone came out to break up the fight.
- top-2 [RED, cos 0.39] (same video): Six or seven people came over and beat up the black man who stole things.
- top-3 [RED, cos 0.38] (same video): The scene was very chaotic. Four or five men still beat the boy violently, and at the same time, someone came to dissuade him.

**13. Assault006_x264**: true zone RED, predicted RED, true sentence ranked #147 of 1968

- GT: *The boy was pushed from the bed to the ground and continued to be beaten until a man in red appeared to stop him.*
- top-1 [RED, cos 0.40]: Three men beat a man with sticks until someone came out to break up the fight.
- top-2 [YELLOW, cos 0.39]: Several people came out of the house next to them to watch or help.
- top-3 [YELLOW, cos 0.38]: A lady came to watch five people fight

**14. Burglary032_x264**: true zone RED, predicted RED, true sentence ranked #94 of 1968

- GT: *Another African-American man wearing a white shirt and a white hat also jumped from the window and entered the room.*
- top-1 [GREEN, cos 0.31]: There is a man squatting against the wall in the upper left corner of the screen
- top-2 [GREEN, cos 0.30]: A man in the distance stood up and walked to the right of the camera
- top-3 [GREEN, cos 0.30]: Another tall man walked into the room

**15. Burglary032_x264**: true zone RED, predicted YELLOW, true sentence ranked #152 of 1968

- GT: *The man ran to the desk in the office again, took a quick look, then ran back, jumped out of the window and left*
- top-1 [YELLOW, cos 0.30]: Repeat The man turned and left the room.
- top-2 [GREEN, cos 0.28]: A man looked around at the door of the house and then walked in and walked to the side of the chair
- top-3 [RED, cos 0.27] (same video): The man walked around the room and stopped at the cabinet and desk inside. He kept rummaging around the desk for items.

**16. Fighting042_x264**: true zone RED, predicted RED, true sentence ranked #107 of 1968

- GT: *The man in the hat came back with a gun and tried to snatch the gun from the security guard*
- top-1 [YELLOW, cos 0.29]: Two more policemen came outside, lingering at the door.
- top-2 [RED, cos 0.27]: The man in black stepped back and took out two guns.
- top-3 [RED, cos 0.24]: Two armed police officers broke into the door.

**17. Arrest024_x264**: true zone RED, predicted RED, true sentence ranked #6 of 1968

- GT: *Police shot at the white car.*
- top-1 [YELLOW, cos 0.31] (same video): Another man in white jumped out of the white car and lay motionless on the ground.
- top-2 [RED, cos 0.28]: A black car was hit by a white car while passing through the alley, and a woman was also knocked to the ground.
- top-3 [YELLOW, cos 0.26] (same video): The black car parked behind the white car and four policemen with guns got out.

**18. RoadAccidents133_x264**: true zone RED, predicted YELLOW, true sentence ranked #18 of 1968

- GT: *A white car collided with a motorcycle while turning*
- top-1 [RED, cos 0.28]: A black car hit two moving motorcycles. One of the riders is wearing blue clothes and one rider is wearing purple clothes
- top-2 [RED, cos 0.27]: A blue car collided with a yellow car
- top-3 [GREEN, cos 0.26]: A blue electric car drove along the road

**19. Fighting047_x264**: true zone RED, predicted RED, true sentence ranked #1 of 1968

- GT: *Several people came and beat each other in the hall again. Two people were knocked down and then pulled up.*
- top-1 [RED, cos 0.27] **(the GT sentence)**: Several people came and beat each other in the hall again. Two people were knocked down and then pulled up.
- top-2 [RED, cos 0.26]: Two armed police officers broke into the door.
- top-3 [GREEN, cos 0.23]: The police at the door stabilized the innocent people inside the house

**20. Shoplifting017_x264**: true zone RED, predicted RED, true sentence ranked #80 of 1968

- GT: *The man in the gray suit put the bag on the table, then reached for something on the counter;*
- top-1 [GREEN, cos 0.29]: The woman took off the earrings with one hand and handed them to the man. A customer in the store left the store
- top-2 [YELLOW, cos 0.28]: The man put down the cash register
- top-3 [GREEN, cos 0.26]: The man returned to the counter and handed something to the clerk, who gave him something

**21. RoadAccidents001_x264**: true zone YELLOW, predicted YELLOW, true sentence ranked #94 of 1968

- GT: *A motorcycle rider got off the car to check the situation of a person wearing blue clothes and a black vest lying on the ground.*
- top-1 [RED, cos 0.35]: The motorcycle caught fire while refueling. People around dispersed, and the man in white tried to put out the fire.
- top-2 [RED, cos 0.35]: An explosion occurred in the distance, and a large amount of thick smoke floated away after a burst of fire.
- top-3 [RED, cos 0.34]: A house in the distance exploded, a large amount of gray smoke filled the air, and many people ran away

**22. Shooting015_x264**: true zone YELLOW, predicted RED, true sentence ranked #188 of 1968

- GT: *A man pressed against the security guard’s neck*
- top-1 [RED, cos 0.36]: The man took out his gun and shot at the door
- top-2 [YELLOW, cos 0.32]: The security guard took out his pistol and looked outside the door
- top-3 [YELLOW, cos 0.31]: The man in the hat turned back outside the door twice and ran away

**23. Robbery050_x264**: true zone YELLOW, predicted RED, true sentence ranked #90 of 1968

- GT: *A man in a striped shirt ran out from behind the white car, grabbed something from the car and then went around to the other side.*
- top-1 [GREEN, cos 0.34]: A white man with black hair and a beard got out of a white car on the roadside, wearing blue overalls.
- top-2 [GREEN, cos 0.31]: A man in a hat and white clothes is wandering in front of a white van
- top-3 [GREEN, cos 0.31]: A man in white walked from a white car to a roadside shop.

**24. Fighting003_x264**: true zone YELLOW, predicted RED, true sentence ranked #32 of 1968

- GT: *Two men at the back of the screen run towards the center*
- top-1 [RED, cos 0.31]: Five people fighting with each other
- top-2 [RED, cos 0.28]: Three men beat a man with sticks until someone came out to break up the fight.
- top-3 [RED, cos 0.28]: Six or seven people came over and beat up the black man who stole things.

**25. Arson022_x264**: true zone YELLOW, predicted YELLOW, true sentence ranked #106 of 1968

- GT: *The man came to a Christmas tree in the right corner of the house*
- top-1 [RED, cos 0.35]: A man in black walked into the counter, opened the cabinet and rummaged through something
- top-2 [GREEN, cos 0.32]: A man in black walked through the corridor
- top-3 [RED, cos 0.31]: The man walked around the room and stopped at the cabinet and desk inside. He kept rummaging around the desk for items.

**26. Arson041_x264**: true zone YELLOW, predicted GREEN, true sentence ranked #9 of 1968

- GT: *A man in black was wandering in front of the door, standing,*
- top-1 [GREEN, cos 0.30]: The man in white walked to the gap between the two houses
- top-2 [YELLOW, cos 0.30]: The man in the hat turned back outside the door twice and ran away
- top-3 [GREEN, cos 0.29]: The man who had packed his things stood up, turned around and walked upwards

**27. Abuse030_x264**: true zone YELLOW, predicted YELLOW, true sentence ranked #1084 of 1968

- GT: *The man begins to take the belt*
- top-1 [YELLOW, cos 0.30]: The man stood up while observing
- top-2 [GREEN, cos 0.29]: There is a white old man nearby crossing the road and walking towards the distance
- top-3 [GREEN, cos 0.29]: There is a white old man nearby crossing the road and walking into the distance

**28. Fighting033_x264**: true zone YELLOW, predicted GREEN, true sentence ranked #1499 of 1968

- GT: *The couple stopped and looked back to observe*
- top-1 [RED, cos 0.27]: The man standing in front of the door kicked hard at the door.
- top-2 [GREEN, cos 0.25]: A man in black with short hair stood in the hall with his arms akimbo
- top-3 [GREEN, cos 0.24]: A man in blue waving at the door

**29. Burglary033_x264**: true zone YELLOW, predicted YELLOW, true sentence ranked #149 of 1968

- GT: *A young white man wearing a black coat with a white pattern on the back was groping in front of the door and trying to open it.*
- top-1 [RED, cos 0.33]: A man entered the elevator and kept lighting the camera on a corner of the elevator with a lighter.
- top-2 [YELLOW, cos 0.27]: Two policemen opened the door to the room and shined flashlights on the man.
- top-3 [GREEN, cos 0.25]: There is an old man holding something in front of the camera

**30. Shoplifting044_x264**: true zone YELLOW, predicted GREEN, true sentence ranked #386 of 1968

- GT: *Three men looked at the computer. The man in black clothes behind looked around, and then walked behind the man in the brown shirt.*
- top-1 [GREEN, cos 0.33]: Black-haired female clerk introduces products to a black-haired man
- top-2 [GREEN, cos 0.30]: Black-clad supermarket attendant is chatting with customers
- top-3 [GREEN, cos 0.29]: Two people came to the counter to chat with the service staff to handle business

**31. Shoplifting037_x264**: true zone YELLOW, predicted YELLOW, true sentence ranked #18 of 1968

- GT: *The woman shouted in the direction of the door and followed her*
- top-1 [GREEN, cos 0.31]: A child came in at the door
- top-2 [GREEN, cos 0.30]: The woman in red came to the shelf and took down a tube
- top-3 [GREEN, cos 0.26]: The mother of the little girl at the door came to the counter to call someone

**32. Shooting032_x264**: true zone YELLOW, predicted YELLOW, true sentence ranked #204 of 1968

- GT: *Both of them walked away*
- top-1 [GREEN, cos 0.27]: A woman in black walked away along the road
- top-2 [GREEN, cos 0.26]: A few men in black got out of the car
- top-3 [GREEN, cos 0.24]: A man in black walked by the side of the road

**33. Shooting032_x264**: true zone YELLOW, predicted RED, true sentence ranked #1214 of 1968

- GT: *One of the staff entered the arch*
- top-1 [YELLOW, cos 0.34] (same video): Three people surrounded the man on the ground
- top-2 [YELLOW, cos 0.33]: Seven people walked to the site of the car accident to check on the injured
- top-3 [YELLOW, cos 0.32] (same video): Four people look at the man on the ground

**34. Shooting032_x264**: true zone YELLOW, predicted RED, true sentence ranked #890 of 1968

- GT: *The man in a yellow shirt walked up to him and looked at him*
- top-1 [YELLOW, cos 0.39] (same video): Three people surrounded the man on the ground
- top-2 [YELLOW, cos 0.37]: Three men were lying on the ground.
- top-3 [YELLOW, cos 0.35]: Pedestrians on the roadside helped two people who fell to the ground

**35. Burglary032_x264**: true zone YELLOW, predicted RED, true sentence ranked #138 of 1968

- GT: *The man in white clothes shook his head and walked away*
- top-1 [GREEN, cos 0.32]: Another tall man walked into the room
- top-2 [GREEN, cos 0.30]: A man in the distance stood up and walked to the right of the camera
- top-3 [GREEN, cos 0.28]: The man stood up and squatted down again.

**36. Shoplifting010_x264**: true zone YELLOW, predicted GREEN, true sentence ranked #377 of 1968

- GT: *The woman carrying a purple schoolbag, after taking snacks from the shelf, walked to the door, then turned back and wandered twice, and then walked out of the screen*
- top-1 [GREEN, cos 0.25]: The man in red clothes walked from the house to the front of the man at the counter
- top-2 [GREEN, cos 0.24]: A man in red clothes walked out to the street.
- top-3 [GREEN, cos 0.24]: The man in red clothes returns to the left side of the counter

**37. Stealing079_x264**: true zone YELLOW, predicted YELLOW, true sentence ranked #477 of 1968

- GT: *The man in white dragged the shovel across the motorcycle and walked to the middle of the road and looked into the distance*
- top-1 [RED, cos 0.28]: The man opened the drawer and took out cash and some items.
- top-2 [YELLOW, cos 0.28]: The man pulled the bag in his hand open a little, then closed it and walked out for a few steps, then walked back to the chair and placed the bag on the chair
- top-3 [GREEN, cos 0.28]: The man sitting on the ground got up and walked to the counter. He put the packaged goods on the counter and then turned back and sat down.

**38. Fighting047_x264**: true zone YELLOW, predicted GREEN, true sentence ranked #4 of 1968

- GT: *After two men quarreled with each other, the two groups dispersed*
- top-1 [GREEN, cos 0.28]: Multiple police officers walking back and forth in the room,
- top-2 [RED, cos 0.27] (same video): Several people came and beat each other in the hall again. Two people were knocked down and then pulled up.
- top-3 [RED, cos 0.26]: Five people fighting with each other

**39. Shooting010_x264**: true zone YELLOW, predicted RED, true sentence ranked #111 of 1968

- GT: *A man also came to the car*
- top-1 [GREEN, cos 0.35]: There is a white car parked in front of the camera
- top-2 [GREEN, cos 0.32]: A white car drove into the garage
- top-3 [YELLOW, cos 0.32]: A yellow electric car came and parked next to it

**40. Vandalism007_x264**: true zone YELLOW, predicted RED, true sentence ranked #46 of 1968

- GT: *The man wearing a hat and carrying a black bag walked to the black car parked on the roadside, touched the rearview mirror and backed away*
- top-1 [GREEN, cos 0.32]: A man wearing a white hat passed by the road
- top-2 [GREEN, cos 0.29] (same video): A man wearing a hat and carrying a black bag walked to a car parked on the street
- top-3 [GREEN, cos 0.28]: A white car passed by the small road of the manor

## 7. Next steps

1. **Pick the Head A operating point deliberately.** Add a softer RED-recall floor (e.g. 0.6) or extend the threshold grid below
   0.05, and report a precision/recall curve for RED instead of a single point.
2. **Better retrieval:** temporal context (neighbouring sentences / 32-frame clips), hard-negative mining within the same video,
   and evaluating against *distinct* sentence texts only, which is fairer for templated captions.
3. **Unfreeze a little:** fine-tune the last V-JEPA blocks or train a token-level attention pool instead of mean pooling.
4. **Video-level fusion:** combine sentence-level zones over time (max RED probability over a sliding window) and compare
   with phase-1 Head A on video and frame AUC.

## Files

| What | Where (cluster: `/projects/adsum-mllm-om/ucf_anomaly/`) |
|---|---|
| Code | `poc/align.py` (text / clips / train / eval), `poc/align.sh`, `poc/zone_heads.py` |
| Labels | `uca_labels/sentence_labels.jsonl` |
| Text embeddings | `text_emb/sentences.npy` (+ `sentences_ids.npy`), API cache in `text_emb/cache/` |
| Clip embeddings | `vjepa_emb/part_00..47.npz` (X, ids), merged into `vjepa_emb/sentence_emb.npz` |
| Checkpoints | `heads/align_head.pt` (alignment + zone branch, with input normalisation), `poc/heads/head_a_zone.pt` (with thresholds), `poc/heads/head_b_{13,grouped}.pt` |
| Results | `results/phase2_eval.json`, `results/phase2_train.json`, `poc/results/zone_heads.json`; logs in `logs/align_*` and `poc/logs/zone_heads_*` |

### Inference (repo copy)

`poc/demo2.py` runs the live pipeline from `inference_flowchart.excalidraw` (`flowchart.png` at the repo root):
- Preprocess (4:3 crop, 224 px, 7.5 fps) -> rolling 16-frame clips every 8 frames -> V-JEPA 2 (fp16).
- **Head A** (`heads/head_a_zone.pt`) -> mean of the last 4 clips + hysteresis (t = 0.5, exit at t - 0.1).
- YELLOW/RED: **Head C** (`heads/head_c_explain.pt`, the alignment head, copied from `heads/align_head.pt`) -> top-3 phrases from the standard dictionary (`dictionary/`, 255 concepts) = explanation + severity.
- RED only: **Head B** (`heads/head_b_13.pt`) gives the type if its top-1 probability is >= 0.6, else "Unclassified incident".
- Alerts are deduplicated within 10 s (escalating yellow -> red), and each gets a 4 s clip with faces blurred.

V-JEPA 2 is frozen, not fine-tuned as the flowchart says. `heads/head_b_grouped.pt` (6 groups) is an unused alternative to `head_b_13.pt`.
Runs at ~250 ms/clip on an RTX 3050 laptop GPU. Snapshot: `examples/out/phase2_preview_grid.jpg`.
