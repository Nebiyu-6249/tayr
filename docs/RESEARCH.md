# Tayr — Phase 0 Research

**Accessed / executed:** 2026-08-25. All web sources fetched on that date unless noted.
**Status:** Phase 0 deliverable. No source code has been written. Awaiting approval per §2.2.

---

## 0. How to read this document

Every non-trivial external claim carries one of three markers, per the project's §1.2 rule:

- `[VERIFIED: <url or command>]` — checked against a primary source in this session.
- `[ASSUMED]` — believed, not checked, and cheap to be wrong about.
- `[UNKNOWN]` — not known, and not guessed.

**Primary source** here means the project's own repository, its package-index metadata, or its
generated SDK source. Blog posts, aggregator sites and AI-summary sites are *not* primary and are
labelled as secondary wherever used.

One concrete illustration of why this matters, from this session: a widely-cited AI-summary site
stated the DUT Anti-UAV dataset is MIT-licensed. The repository itself says **Apache-2.0**.
The summary was wrong. See §5.1.

### Verification blocked by this environment

The container's egress proxy blocked several vendor domains outright, so some claims that would
normally be easy to verify are marked `[UNKNOWN]` rather than guessed:

```
download.pytorch.org -> 000 (blocked)
platform.openai.com  -> 000 (blocked)
openai.com           -> 000 (blocked)
docs.ultralytics.com -> 000 (blocked)
nextjs.org           -> 000 (blocked)
runpod.io            -> 000 (blocked)
lambdalabs.com       -> 000 (blocked)
vast.ai              -> 000 (blocked)
modal.com            -> 000 (blocked)
pypi.org             -> 200 (reachable)
registry.npmjs.org   -> 200 (reachable)
github.com           -> reachable via WebFetch
```
`[VERIFIED: curl -o /dev/null -w '%{http_code}' per host, run 2026-08-25]`

Consequences: **OpenAI pricing could not be verified and is `[UNKNOWN]`** (§10). **GPU host pricing
could not be verified against vendor pages and is secondary-source only** (§11).

---

## 1. This build environment

`[VERIFIED: python3 --version; node --version; docker --version; nvidia-smi; ffmpeg -version; git log]`

| Fact | Value | Consequence |
|---|---|---|
| Python | 3.11.15 | Below the 3.12 floor several current libs now require — see §9 |
| Node | v22.22.2 | Satisfies Next.js 16 (`>=20.9.0`) |
| Docker | 29.3.1 | Compose workflow viable |
| **GPU** | **none — `nvidia-smi: command not found`** | **No training, no real metrics here. See §12 Risk R1** |
| ffmpeg | not installed | Must come from the container image |
| Repo | zero commits, empty tree | Clean start; `.gitignore` lands in commit 1 |

---

## 2. The licence problem, and why it drives the architecture

This is the single most consequential finding in Phase 0, and it changes the default stack the
brief proposed.

### 2.1 Ultralytics is AGPL-3.0

`[VERIFIED: https://github.com/ultralytics/ultralytics — sidebar "AGPL-3.0 License"]`
`[VERIFIED: https://pypi.org/pypi/ultralytics/json — "license": "AGPL-3.0", version 8.4.128]`

The README frames AGPL-3.0 as "perfect for students, researchers, and enthusiasts", and offers a
paid Enterprise Licence for "bypassing the open-source requirements of AGPL-3.0".
`[VERIFIED: https://github.com/ultralytics/ultralytics]`

What AGPL-3.0 means for Tayr specifically: AGPL's §13 network clause is triggered by *serving the
software over a network*, which is exactly what the visitor mode in §3.2 does. The copyleft reaches
the whole derivative work — the FastAPI app, the worker, the frontend, config, and arguably the
trained weights. For a **public** portfolio repo this is satisfiable (publish everything under
AGPL-3.0), but it has real downstream cost:

- The entire Tayr codebase must be AGPL-3.0. Not MIT, not Apache-2.0.
- Any future commercial use, or any employer wanting to build on it, needs an Ultralytics
  Enterprise Licence or a rewrite.
- Some employers' legal teams treat AGPL in a candidate's portfolio repo as a flag.

The scope of copyleft over *model weights* specifically is contested and I am not going to state it
as settled. `[UNKNOWN]` — treat as unresolved and avoid the question by not using Ultralytics.

### 2.2 BoxMOT is also AGPL-3.0

`[VERIFIED: https://github.com/mikel-brostrom/boxmot — "AGPL-3.0" licence]`

The obvious batteries-included tracker library carries the same problem. It bundles OccluBoost,
BoTSort, BoostTrack, StrongSort, DeepOCSort, ByteTrack, HybridSort, OCSort, SFSort, and supports
Python 3.10–3.13. `[VERIFIED: same]`

### 2.3 Permissively-licensed alternatives that do exist

| Component | Licence | Evidence |
|---|---|---|
| **SAHI** (slicing) | MIT | `[VERIFIED: https://github.com/obss/sahi; pypi sahi 0.12.6 license MIT]` |
| **RF-DETR** (detector) | Apache-2.0 for Nano/Small/Medium/Large + all segmentation; PML-1.0 for XL/2XL | `[VERIFIED: https://github.com/roboflow/rf-detr]` |
| **D-FINE** (detector) | Apache-2.0 (Objects365-trained ckpts subject to that dataset's terms) | `[VERIFIED: https://github.com/Peterande/D-FINE]` |
| **YOLOX** (detector) | Apache-2.0 | `[VERIFIED: https://github.com/Megvii-BaseDetection/YOLOX]` |
| **ByteTrack** (reference impl) | MIT | `[VERIFIED: https://github.com/FoundationVision/ByteTrack]` |
| **BoT-SORT** (reference impl) | MIT | `[VERIFIED: https://github.com/NirAharon/BoT-SORT]` |
| **torchvision** | BSD | `[VERIFIED: pypi torchvision 0.28.0]` |

### 2.4 Recommendation

**Build permissive. Do not take the Ultralytics dependency.**

- Detector: **D-FINE** or **RF-DETR**, both Apache-2.0. See §3.3 for which and why.
- Slicing: **SAHI** (MIT) — it supports both HuggingFace RT-DETR-family and Roboflow/RF-DETR
  natively. `[VERIFIED: https://github.com/obss/sahi]`
- Tracking: implement ByteTrack association ourselves against the MIT reference, rather than
  depending on AGPL BoxMOT. The association step is small and we need to tune it for aerial
  targets anyway (§4).
- Licence the repo **Apache-2.0**.

If you would rather use Ultralytics for velocity and accept AGPL for the whole project, that is a
legitimate call — but it should be a deliberate decision recorded in `CLAUDE.md`, not a default
that arrives by accident through `pip install ultralytics`. **This is decision D1 in §13.**

---

## 3. Small-object detection

### 3.1 What a P2 head is and what it costs

A standard YOLO-family detect head runs at P3/P4/P5 — strides 8, 16, 32. A P2 head adds a
stride-4 branch taken from an early, high-resolution backbone feature map. The practical effect is
that the smallest reliably-detectable object shrinks: the P2 branch is reported to help most in
roughly the 6×6 to 16×16 pixel range, which P3 handles poorly because of accumulated downsampling.
*(secondary — search synthesis over Ultralytics discussion #8227 and several 2025–2026 arXiv papers;
I have not verified any single number here against a primary source.)*

The cost is FLOPs and activation memory, and it lands at the *highest* resolution in the network,
so it is the most expensive place to add compute. A commonly-reported mitigation is dropping the P5
head when the task has no large objects — which is exactly Tayr's situation, since a drone at range
is never large. *(secondary, same caveat.)*

For Tayr this is a genuinely good trade: we have no large-object class worth spending P5 on.
**Concrete plan: measure P2-on/P5-off against the stock head as an ablation in Phase 3, and report
both accuracy and ms/frame.** Do not assume it helps; the brief is right to demand it be measured.

### 3.2 SAHI / sliced inference

SAHI slices a large frame into overlapping tiles, runs the detector on each tile at native
resolution, and merges detections back to global coordinates. It is model-agnostic and currently
integrates with Ultralytics, MMDetection, HuggingFace (RT-DETRv2, RT-DETR, GroundingDINO),
Torchvision, YOLOv5, Detectron2, Roboflow/RF-DETR, and YOLOX.
`[VERIFIED: https://github.com/obss/sahi]`

The cost is the thing to be honest about: SAHI multiplies inference passes per frame. A 3840×2160
frame at 640×640 tiles with 20% overlap is on the order of 42 tiles, so a single frame costs ~42
forward passes plus merge/NMS post-processing. At 30fps that is ~1260 forward passes per second of
video. *(arithmetic is mine and exact; the claim that SAHI multiplies passes is
`[VERIFIED: https://github.com/obss/sahi]` by construction of the method.)*

Published AP gains for SAHI are quoted in the 6.8–14.5% range *(secondary; not verified against the
SAHI paper, and the figure is dataset- and detector-dependent — do not cite this in the writeup
without checking the source)*.

**This is why §4.1's region-proposal stage exists, and it is the right instinct.** Running 42 tiles
over empty sky is waste. Ego-motion-compensated frame differencing can cut the tile count by an
order of magnitude on typical footage by proposing only regions containing motion. But it must be
measured, not assumed — and it introduces a recall ceiling, because anything the proposer misses
the detector never sees. **Plan: instrument recall of the proposer independently, and report the
proposer's own miss rate as a first-class metric.** A proposal stage that silently drops 5% of
targets would otherwise be invisible in end-to-end mAP and would look like a detector failure.

### 3.3 Detector choice: D-FINE vs RF-DETR

Both are Apache-2.0 and both are current.

- **D-FINE**: five sizes N/S/M/L/X, COCO-only checkpoints from 42.8% to 55.8% AP, Objects365+COCO
  variants 50.7%–59.3% AP, and Objects365-pretrained weights explicitly recommended for transfer
  learning. Stated `python=3.11.9`. `[VERIFIED: https://github.com/Peterande/D-FINE]`
- **RF-DETR**: DINOv2 backbone, Nano→Large under Apache-2.0, `Python>=3.10`, resolutions from
  312×312 to 880×880, and it is on PyPI as `rfdetr` 1.9.4 published 2026-08-24 — i.e. actively
  shipping. `[VERIFIED: https://github.com/roboflow/rf-detr; https://pypi.org/pypi/rfdetr/json]`

**Recommendation: RF-DETR**, on three grounds — it is packaged and installable (D-FINE is a
research repo you vendor), it is under active release, and SAHI lists it as a supported backend.
Keep the detector behind an interface so D-FINE can be swapped in for an ablation. Note the DETR
family does not have a "P2 head" in the YOLO sense; the equivalent knob is input resolution and
backbone feature selection. **If the P2 ablation in §3.1 is important to you as a research
contribution, that argues for a YOLO-family architecture (YOLOX, Apache-2.0) instead — this is
decision D2 in §13.**

---

## 4. Tracking

| | ByteTrack | BoT-SORT |
|---|---|---|
| Licence | MIT `[VERIFIED]` | MIT `[VERIFIED]` |
| Repo activity | 389 commits `[VERIFIED]` | 26 commits `[VERIFIED]` |
| Stated deps | requirements.txt, pycocotools, cython_bbox `[VERIFIED]` | Python 3.7, PyTorch 1.11.0+cu113, torchvision 0.12.0, faiss, FastReID `[VERIFIED]` |
| Core idea | Associate *every* detection box, including low-score ones, by matching them to existing tracklets `[VERIFIED]` | Motion + appearance + camera-motion compensation + improved KF state vector `[VERIFIED]` |

Sources: `https://github.com/FoundationVision/ByteTrack`, `https://github.com/NirAharon/BoT-SORT`.

**For Tayr's stated failure mode — targets that intermittently drop below detection threshold —
ByteTrack's design is the direct answer.** That is precisely the case it was built for: it keeps
low-confidence boxes and resolves them against existing tracklets instead of discarding them. A
drone fading to 12 pixels against bright sky is a low-score detection, not an absent one.

BoT-SORT's appearance/ReID branch is the part that will *not* transfer: ReID embeddings are learned
on pedestrian crops with rich texture. A 15-pixel grey blob has almost no appearance signal — which
is, after all, the project's own hypothesis. Its camera-motion compensation *is* relevant and worth
borrowing separately.

Its pinned stack (Python 3.7, torch 1.11) is six years stale and cannot be installed alongside the
rest of this project. Do not depend on that repo; read it.

**Recommendation:** implement ByteTrack-style association in-project (MIT reference to hand),
plus a constant-velocity Kalman filter, plus BoT-SORT's camera-motion-compensation idea.

**On the Kalman filter:** `filterpy` — the usual Python KF package — last released **1.4.5 on
2018-10-10** `[VERIFIED: https://pypi.org/pypi/filterpy/json]`. That is nearly eight years dead.
A constant-velocity KF for a 2D box is ~40 lines of NumPy. **Write it, do not depend on filterpy.**
It also lets us tune Q/R for aerial motion, which we need to do regardless.

---

## 5. Datasets

### 5.1 Verified per-dataset findings

**DUT Anti-UAV** — `[VERIFIED: https://github.com/wangdongdut/DUT-Anti-UAV]`
- Licence: **Apache-2.0** (repo sidebar). *An AI-summary site claimed MIT. It was wrong.*
- Download: Google Drive and Baidu links for detection train/val/test and tracking img/gt, listed in
  the README.
- Citation required: Zhao, Zhang, Li, Wang, *Vision-based Anti-UAV Detection and Tracking*,
  IEEE T-ITS 2022.
- Split sizes: **not stated in the README.** Secondary sources say 10,000 detection images split
  5,200 / 2,600 / 2,200 and 20 tracking videos — `[UNKNOWN]`, to be confirmed by downloading and
  counting. Do not put the secondary numbers in the dissertation.
- Annotation format: not stated in README. `[UNKNOWN]` until download.

**Drone-vs-Bird / WOSDETC** — `[VERIFIED: https://github.com/wosdetc/challenge]`
- Access: email `wosdetc@googlegroups.com`; **"You will be asked to sign a data usage agreement"**,
  after which you "can then use the data for research purposes".
- **No redistribution rights are granted.** This is the DUA dataset the brief's §5.3 warned about.
  Nothing derived from it — frames, crops, converted annotations — goes in the public repo.
- Annotation format is stated exactly:
  `framenum num_objs_in_frame obj1_x_left obj1_y_top obj1_w obj1_h obj1_class ...`
  One text file per video. **Note it carries a class field.**
- Size: not disclosed; "continually increased over consecutive installments".
- **Lead time is a schedule risk** — a human has to approve the DUA. Request it on day one.

**Anti-UAV (Jiang et al.)** — `[VERIFIED: https://github.com/ZhaoJ9014/Anti-UAV]`
- Licence: **MIT** ("The project of Anti-UAV is released under the MIT License").
- **Critical: "410 and 600 versions only contain IR videos while 300 version contains both RGB
  videos and IR videos."** Only Anti-UAV300 is usable for an RGB pipeline.
- Annotated with bounding boxes, attributes, and per-frame target-exists flags.

**LRDDv2** — `[VERIFIED: https://arxiv.org/abs/2508.03331; https://research.coe.drexel.edu/ece/imaple/lrddv2/ via search]`
- Access: fill in a form; **the university clears you against US export-control regulations** before
  emailing a link. *(secondary — the Drexel page itself was not fetched.)*
- Size: 39,516 annotated images, range information for 8,000+, majority ≤50 px in 1080p. *(secondary,
  from the arXiv abstract listing.)*
- **Export-control clearance is an unbounded-latency dependency and may simply be refused.** Treat
  LRDDv2 as optional. Do not put it on the critical path.

**Det-Fly** — 13,271 images at 3840×2160, air-to-air micro-UAV, repo cited as
`https://github.com/Jake-WU/Det-Fly`. *(secondary; repo not fetched, licence `[UNKNOWN]`.)*

**MAV-VID** — 64 videos / 40,232 images of single drones; distributed via Kaggle. *(secondary;
licence `[UNKNOWN]`.)*

**NPS-Drones** — used as a small-drone benchmark. Access, size and licence all `[UNKNOWN]`.

**UAVDetectionTrackingBenchmark** — `[VERIFIED: https://github.com/KostadinovShalon/UAVDetectionTrackingBenchmark]`
Directly useful: it already ships COCO converters — `convert_mav_vid_to_coco.py`,
`convert_drone_vs_bird_to_coco.py`, `convert_anti_uav_to_coco.py`, `video_to_images.py` — and
publishes mean target sizes that matter enormously for Tayr:

| Dataset | Split | Mean target size |
|---|---|---|
| MAV-VID | 53 train / 11 val videos | **215×128 px** |
| Drone-vs-Bird | 61 train / 16 val videos | **34×23 px** |
| Anti-UAV RGB | 60 train / 40 val videos | 125×59 px |
| Anti-UAV IR | — | 52×29 px |

**Licence of that repo is not stated** `[VERIFIED: not present on the repo page]` — so treat the
converters as *reference to read*, not code to copy, until the licence is established.

Note what that table says about the brief's pixel buckets. Only Drone-vs-Bird lives near the
<32px regime the research question targets. MAV-VID at 215×128 is not a small-object dataset at all.

### 5.2 The class-label problem — this blocks §4.3 as written

**This is the most serious internal inconsistency in the brief, and it needs resolving before
Phase 2.**

§4.3 requires a "per-class confusion matrix (drone / bird / aircraft)". §4.1 stage 6 requires a
drone/bird/aircraft/unknown classifier. But of the seven datasets specified:

| Dataset | Classes actually available |
|---|---|
| DUT Anti-UAV | UAV only |
| Anti-UAV (Jiang) | UAV only |
| LRDDv2 | drones only |
| MAV-VID | single drones |
| NPS-Drones | drones |
| Det-Fly | MAVs only |
| Drone-vs-Bird | has a class field in the annotation format `[VERIFIED]`; whether birds are *annotated as a class* or merely present as unlabelled distractors is `[UNKNOWN]` |

**Six of the seven are single-class drone datasets.** You cannot train a three-class classifier on
them, and you cannot produce the required confusion matrix. At best you get drone-vs-background.

Candidate fixes found. **Both were investigated further after D3 was decided; see §14.2 — the
verification could not be completed from this container, and AOD-4 has two problems beyond licence.**

- **AOD-4** — 22,516 images across four classes: airplanes, helicopters, drones, birds; ~7,900
  annotations per class. Hosted on Mendeley Data, DOI `10.17632/cd5z895tr2.1`, direct URL
  `https://data.mendeley.com/datasets/cd5z895tr2/1`. *(secondary:
  `https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11372627/`,
  `https://www.sciencedirect.com/science/article/pii/S2352340924007650`.)*
  **Licence `[UNKNOWN]` — every route to it is egress-blocked from this container (§14.2).**
- **YOLOBirDrone** — bird/drone bounding boxes with class labels. *(secondary:
  `https://arxiv.org/html/2601.08319v1`.)* Licence and availability `[UNKNOWN]`; `arxiv.org` is
  egress-blocked here.

**Decision D3 is resolved in principle (§14.1) but its precondition is unmet. See §14.2.**

### 5.3 The harder problem: the hypothesis needs *tracks*, and there are very few

This is subtler than the class problem and I think it is the real threat to the research.

§4.1 stage 5 extracts motion features over a sliding window — velocity profile, acceleration
variance, vertical-oscillation FFT, heading entropy, hover duration. Every one of those requires a
**contiguous track**, not a frame. The training unit for the §4.1-stage-6 classifier is therefore
**one track**, not one image.

Now count the actual supply:
- DUT Anti-UAV's *detection* subset is 10,000 **still images**. Worth **zero** tracks.
- DUT Anti-UAV's *tracking* subset is ~20 videos. That is on the order of **20 drone tracks.**
- Anti-UAV300 gives 60 train / 40 val **videos**. `[VERIFIED via benchmark repo]`
- Drone-vs-Bird gives 61 train / 16 val **videos**. `[VERIFIED via benchmark repo]`

So the headline "10,000 images" is misleading for this project's actual purpose. The realistic
order of magnitude for track-level training data is **low hundreds of drone tracks and an unknown,
probably much smaller, number of bird tracks.**

A gradient-boosted tree on ~8 hand-designed features can work at that scale — which is a good
argument for the brief's instinct to start with trees rather than a GRU. But it also means:
- **Cross-validation must be grouped by source video**, or frames from one track leak across the
  split and the reported accuracy is fiction.
- A 1D CNN/GRU (§4.1 stage 6's stretch goal) is very unlikely to be trainable at this scale.
  Expect the tree baseline to be the final answer, and say so honestly if it is.
- Confidence intervals on the final hypothesis test will be **wide**. Plan to report them. A point
  estimate from ~100 tracks with no interval would be the kind of overclaim §1.3 exists to prevent.

**Recommendation: make "count the actual usable tracks per class" the first task in Phase 2, before
writing any converter.** If the answer is under ~50 bird tracks, the hypothesis test as specified is
not adequately powered and the project's framing needs to change — better to learn that in week 2
than in week 20.

---

## 6. Evaluation: two problems with §4.3 as written

### 6.1 mAP@0.5 is close to meaningless in the <8px bucket

IoU is extremely sensitive to small positional error on small boxes. On an 8×8 box, a 2-pixel
offset in one axis drops IoU to roughly 0.6; 3 pixels drops it below 0.5 and the detection is scored
as a miss even though a human would call it a hit. *(The sensitivity direction is well established
— see the Tiny Object Detection Challenge, `https://arxiv.org/pdf/2009.07506`, which set IoU
thresholds of 0.25 and 0.5 precisely because 0.5 is too strict for tiny objects. The specific
numbers in my sentence are my own arithmetic on an idealised square box, not quoted from a paper.)*

Worse, annotation noise is on the same scale as the object. Two competent human annotators will
disagree by 1–2 px on an 8px target, so the metric's noise floor is comparable to its signal.

**Recommendation:** in the <16px buckets, additionally report mAP@0.25, and report detection
counts/recall at a fixed operating point rather than leaning on mAP alone. State the protocol
explicitly next to every number. Normalised Wasserstein Distance is the literature's alternative to
IoU for this regime *(secondary)* — worth evaluating but not worth blocking on.

### 6.2 The hypothesis test as specified does not compare like with like

§4.3 says: *"If the motion-based classifier does not outperform appearance in the bottom buckets,
the hypothesis is wrong."* But the metrics listed just above it (mAP@0.5, mAP@0.5:0.95) measure
**detection**, while the motion classifier does **track classification**. Those are different
quantities with different units. As written, there is no controlled comparison — you would be
comparing a detector's mAP against a classifier's accuracy and declaring a winner.

**To actually test the stated hypothesis you need a matched pair:**
- **Appearance arm:** crop each track's per-frame boxes → small CNN → per-track class by temporal
  vote.
- **Motion arm:** the same tracks → §4.1 stage-5 features → GBT → per-track class.
- Both evaluated on **the same tracks**, in **the same pixel buckets**, with **the same metric**
  (per-class F1 or balanced accuracy), and **grouped splits** per §5.3.

Then "appearance saturates below ~20px while motion does not" becomes a statement you can actually
falsify: two curves against pixels-on-target, with confidence intervals, and a crossover point (or
no crossover, which is a real and publishable result too).

Without the appearance arm there is no hypothesis test — just a classifier with an accuracy score.
**This changes the Phase 5 deliverable and is decision D4 in §13.**

### 6.3 What §4.3 gets right and should keep

False-alarms-per-hour on drone-free footage is the best idea in the brief's evaluation section. It
needs no annotation, it is the number that decides whether such a system is deployable, and it is
genuinely under-reported in the literature. Keep it, and source the negative footage deliberately
(varied sky conditions, birds, clouds, sensor noise, lens dirt).

The cross-dataset generalisation test is also right, and will likely produce the project's most
interesting number — but note it is only meaningful for the *detector*, since only one dataset has
multi-class labels (§5.2).

---

## 7. Anticipated difficulties

Written down before hitting them, per §2.1.

1. **Annotation format sprawl.** Every dataset differs. Drone-vs-Bird is a bespoke space-delimited
   per-video text format `[VERIFIED]`; DUT's is undocumented; Anti-UAV uses per-frame boxes plus
   exists-flags. The brief's demand for explicit converters with round-trip tests is correct — this
   is where silent coordinate bugs live (0- vs 1-indexed, xywh vs xyxy, top-left vs centre).
2. **Target-exists flags.** Anti-UAV annotates frames where the target is absent `[VERIFIED]`.
   A converter that ignores that flag will emit boxes for absent targets and poison training.
3. **Class imbalance at the pixel level.** A 15px target in a 4K frame is ~0.0027% of pixels.
   Tiling helps by raising the positive fraction per tile, but most tiles are still pure background.
4. **Tile-boundary double-counting.** SAHI's merge step must dedupe targets split across tile seams;
   with overlap, one drone can produce two boxes. Verify the merge, don't trust it.
5. **Memory blowup on 4K tiling.** 42 tiles × batch × activations. Batch the tiles, don't materialise
   them all.
6. **Tracker ID switches on small targets.** Two birds crossing at 10px are nearly indistinguishable
   by appearance; ID switches will corrupt track-level features. Report IDF1/ID-switch counts, not
   just MOTA.
7. **Proposer recall ceiling** (§3.2) — instrument it separately.
8. **Grouped-split leakage** (§5.3) — the single easiest way to accidentally fabricate a good result.
9. **CUDA/PyTorch/Python version conflicts** — see §9; already biting.
10. **Determinism is not fully achievable on GPU** — see §8.

---

## 8. On the "deterministic given a seed" requirement (§4.2)

Bit-exact reproducibility on GPU is **not** generally achievable. Several cuDNN kernels and any
op using atomic accumulation are nondeterministic by design, and results can also vary with
batch size, hardware, and library version. `[ASSUMED — this is well-established, but I could not
verify it against PyTorch's own reproducibility docs because pytorch.org is egress-blocked in this
container; verify before writing it in the dissertation.]`

**What is honestly deliverable:**
- Seed Python/NumPy/PyTorch, set deterministic dataloader worker seeding.
- Enable deterministic algorithms where kernels support it, accepting the speed cost.
- Record seed, git commit, dataset checksums, and full library versions in the run manifest.
- Document that reproduction is exact on identical hardware+versions and close-but-not-bitwise
  otherwise.

Promising more than that in a research writeup would be an overclaim. Flagging it now so the
requirement can be reworded rather than quietly missed.

---

## 9. Pinned version matrix

All versions `[VERIFIED: https://pypi.org/pypi/<pkg>/json, queried 2026-08-25]` and
`[VERIFIED: https://registry.npmjs.org/<pkg>/latest]`.

### 9.1 The conflict that forces a decision

| Package | Latest | requires_python | Newest on py3.11 |
|---|---|---|---|
| numpy | 2.5.2 | **>=3.12** | 2.4.6 |
| xgboost | 3.4.1 | **>=3.12** | 3.2.0 |
| scikit-learn | 1.9.0 | >=3.11 | 1.9.0 |
| av (PyAV) | 18.1.0 | >=3.11 | 18.1.0 |
| torch | 2.13.0 | >=3.10 | 2.13.0 |

**This container is Python 3.11.15, which caps numpy at 2.4.6 and xgboost at 3.2.0.**

**Recommendation: target Python 3.12** in the Docker image. It clears both floors, and torch 2.13.0,
torchvision 0.28.0, PyAV 18.1.0, RF-DETR (`>=3.10`) and D-FINE (`3.11.9`) all accept it. Python 3.13
also works for the libraries checked but adds risk for CUDA-adjacent wheels for no benefit.
**Note the dev container will need rebuilding — do not develop on 3.11 and deploy on 3.12.**

### 9.2 Proposed pins

| Layer | Package | Version | Licence | Note |
|---|---|---|---|---|
| Runtime | Python | 3.12.x | PSF | §9.1 |
| DL | torch | 2.13.0 | Apache-2.0 | CUDA build `[UNKNOWN]` — see §9.3 |
| DL | torchvision | 0.28.0 | BSD | |
| Detector | rfdetr | 1.9.4 | Apache-2.0 | released 2026-08-24 |
| Slicing | sahi | 0.12.6 | MIT | released 2026-08-16 |
| Numerics | numpy | 2.4.6 (py3.11) / 2.5.2 (py3.12) | BSD-3-Clause | |
| Classifier | xgboost | 3.4.1 | Apache-2.0 | needs py>=3.12 |
| Classifier | lightgbm | 4.7.0 | — | alternative to xgboost |
| Classifier | scikit-learn | 1.9.0 | BSD-3-Clause | |
| Video | **av (PyAV)** | 18.1.0 | BSD-3-Clause | **see §9.4** |
| Video | opencv-python | 5.0.0.93 | Apache 2.0 | |
| Fetch | yt-dlp | 2026.8.19 | Unlicense | only if §11.3 accepted |
| API | fastapi | 0.141.1 | MIT | |
| API | pydantic | 2.13.4 | MIT | |
| API | uvicorn | 0.52.4 | BSD-3-Clause | |
| API | python-multipart | 0.0.32 | Apache-2.0 | upload handling |
| Queue | arq | 0.28.0 | MIT | recommended — see §9.5 |
| Queue | celery | 5.6.3 | BSD-3-Clause | alternative |
| Queue | redis | 8.1.0 | MIT | client |
| DB | sqlalchemy | 2.0.52 | MIT | |
| DB | alembic | 1.19.1 | MIT | |
| DB | **psycopg** | 3.3.4 | **LGPL-3.0-only** | see note below |
| Auth | argon2-cffi | 25.1.0 | MIT | Argon2id per §5.1 |
| LLM | openai | 3.3.1 | Apache-2.0 | |
| Eval | pycocotools | 2.0.11 | FreeBSD | |
| Lint | ruff | 0.16.4 | MIT | |
| Types | mypy | 2.3.1 | MIT | |
| Test | pytest | 9.1.1 | MIT | |
| Front | next | 16.3.2 | — | engines: node >=20.9.0 |
| Front | react | 19.2.8 | — | |
| Front | typescript | 7.0.2 | — | |
| Front | Node | 22.x LTS | — | container has 22.22.2 |

**psycopg is LGPL-3.0-only** `[VERIFIED: pypi]`. Normal use as an unmodified library is fine and
does not make Tayr LGPL, but since you are being careful about licences, note it and record it.
An Apache-2.0/MIT alternative exists if you would rather avoid LGPL entirely — worth a look but
not a blocker.

### 9.3 CUDA — unresolved

torch 2.13.0's available CUDA build variants could **not** be verified: `download.pytorch.org` and
`pytorch.org` are both egress-blocked here. **`[UNKNOWN]` — must be checked against
`https://pytorch.org/get-started/locally/` from an unblocked network before the Dockerfile is
written.** Getting this wrong is exactly the "hours of debugging" failure §1.1 warns about, so I am
not guessing an index URL.

### 9.4 Video decoding: PyAV, not ffmpeg-python

| Package | Latest release | Verdict |
|---|---|---|
| `ffmpeg-python` | 0.2.0, uploaded **2019-07-06** | **Abandoned — do not use** |
| `av` (PyAV) | 18.1.0, uploaded **2026-08-12** | **Actively maintained — use this** |

`[VERIFIED: https://pypi.org/pypi/ffmpeg-python/json; https://pypi.org/pypi/av/json]`

This directly answers §2.1's question about which wrapper is currently maintained. PyAV also binds
libav* directly rather than shelling out to an `ffmpeg` binary, which gives finer control over
decode limits — useful for the §5.3 resource-exhaustion defences.

### 9.5 Job queue: arq

Given FastAPI and an async codebase, **arq** (MIT, 0.28.0, Redis-backed, asyncio-native) is the
lightest fit and avoids Celery's configuration surface. Celery 5.6.3 (BSD-3-Clause) is the
conservative choice with a larger ecosystem. Either satisfies §4.6. `[VERIFIED: pypi for both]`
Recommend **arq**; this is a low-stakes reversible decision.

---

## 10. LLM layer

**Model IDs are verified from OpenAI's own SDK source**, which is generated from their OpenAPI spec
— a legitimate primary source that does not require reaching the blocked docs site:

`[VERIFIED: pip download openai (3.3.1); openai/types/shared/chat_model.py]`

Current-generation IDs present in that Literal include: `gpt-5.6-sol`, `gpt-5.6-terra`,
`gpt-5.6-luna`, `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.2`, `gpt-5.1`,
`gpt-5`, `gpt-5-mini`, `gpt-5-nano`, plus the 4.x and o-series lineage.

**Pricing is `[UNKNOWN]`.** `platform.openai.com` and `openai.com` are both egress-blocked from this
container (§0), so I could not verify a single price. §4.5 requires pricing be `[VERIFIED]` against
current docs, and I am not going to satisfy that requirement with a recollection. **Open question
Q1 in §13.**

Design, which does not depend on the pricing answer:
- `LLMProvider` protocol with one `summarise(structured_track_record) -> IncidentSummary` method;
  OpenAI implementation behind it, so the provider is swappable per §4.5.
- Structured output, schema-validated with Pydantic on return; validation failure is an error, not
  a silently-accepted string (§1.4).
- The LLM **never** sees pixels and **never** makes a classification decision — it renders an
  already-decided structured record into prose.
- Token counts logged per call for cost tracking; hard per-user and global caps with a circuit
  breaker (§5.1).
- **Graceful degradation:** summary generation failing must never fail the job. Structured results
  are the deliverable; prose is a nicety.
- Every string reaching a prompt — filename, fetched video title, user note — is untrusted and
  structurally separated from instructions (§5.1). This matters more than usual here because §5.4's
  URL fetcher would inject attacker-controlled metadata directly into a prompt.

---

## 11. Deployment

### 11.1 Netlify cannot host this — confirmed as the brief suspected

No GPU, and function timeouts far below video-processing duration. The frontend (Next.js) can go to
Netlify or Vercel. The API and worker cannot. `[ASSUMED — Netlify's specific current limits were not
verified; their docs were not fetched. The architectural conclusion does not depend on the exact
numbers.]`

### 11.2 GPU hosting — secondary sources only

**All GPU vendor domains are egress-blocked from this container** (§0), so none of the following is
`[VERIFIED]`. These are aggregator-blog figures and **must be checked against vendor pricing pages
before any budget decision:**

*(secondary, 2026 aggregator posts)* RTX 4090 roughly $0.29–0.74/hr depending on provider and
spot-vs-on-demand; A100 80GB roughly $0.60–2.06/hr; H100 roughly $1.49–3.99/hr. Marketplace
providers are cheapest but preemptible — one source describes instances reclaimable on 15 seconds'
notice with no uptime SLA, versus SLA-backed on-demand elsewhere.

Sources: `https://tech-insider.org/runpod-vs-lambda-vs-vast-ai-2026/`,
`https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/`,
`https://jarvislabs.ai/ai-faqs/best-cloud-gpu-providers-2026`.

**Practical recommendation regardless of exact price:** training and the demo have different needs.
Rent a preemptible GPU by the hour for *training* (checkpoint-resume per §4.2 makes preemption
survivable — this is a good reason to take that requirement seriously). For the *portfolio demo*, a
persistent GPU running 24/7 is the dominant cost and is probably not worth it: consider CPU
inference at reduced throughput for visitor jobs, or an on-demand GPU that scales to zero. **Open
question Q2 in §13.**

### 11.3 The URL-fetch feature — recommend cutting it from v1

§5.4 already gives permission to ship upload-only if the fetcher cannot be made safe. **I recommend
taking that option**, for two reasons — one security, one legal.

**Security.** The brief's control list is right but incomplete for the YouTube case specifically.
`yt-dlp` does not fetch media from `youtube.com`; it resolves to media URLs on
`*.googlevideo.com` CDN hosts. So a hostname allowlist containing only `youtube.com` does not
describe what the process actually connects to, and an allowlist broad enough to work covers a large
rotating CDN surface. The DNS-rebinding re-check-after-redirect requirement becomes substantially
harder to enforce against that. `[ASSUMED — the googlevideo.com resolution behaviour is
well-known but I did not verify it this session; yt-dlp's own docs were not fetched. Verify before
relying on this either way.]`

**Legal.** Downloading YouTube content is contrary to YouTube's Terms of Service. That is
independent of whether the SSRF controls work, and it sits in a **public** portfolio repository.
`[ASSUMED — YouTube's ToS was not fetched this session.]`

An SSRF hole or a ToS violation in a public repo is worse for a career than a missing convenience
feature — which is exactly the reasoning §5.4 itself gives. **Recommend: file upload only for v1;
revisit deliberately later.** This is decision D5 in §13.

---

## 12. Security notes carried into design

The §5 checklist is largely sound and I am not going to restate it. Four observations:

**R1 — no GPU here.** Phases 3–5 cannot produce real numbers in this container. Everything before
that (converters, round-trip tests, pipeline structure, API, security layer) is fully buildable and
CPU-testable. Any figure produced here before a GPU is rented would be either synthetic or absent —
and per §1.3 synthetic fixtures will be labelled as such in code, logs and UI.

**`torch.load` is RCE** — the brief is right. `weights_only=True` or safetensors for all internal
loads; never load a user-supplied checkpoint; checksum repo checkpoints. `[ASSUMED — the pickle
behaviour is well-established; PyTorch's own docs were not reachable to verify the current default.]`

**"Encrypt sensitive columns at rest" (§5.1) is close to a no-op here.** The only sensitive columns
in this schema are email and password hash. A password hash must *not* be encrypted — it is already
a one-way hash, and adding a reversible layer with a key on the same host is worse, not better.
Recommend: volume-level encryption, plus column encryption only if PII scope grows. Record as an
accepted deviation in the threat model rather than implementing security theatre.

**Media decoding sandbox is the highest-value control in §5.** Attacker-supplied video into
libav* is the genuine RCE surface. PyAV (§9.4) helps because decode limits can be enforced in-process
before full decode, satisfying §5.3's "probe the container before full decode" requirement.

---

## 13. Open questions and risks

### Decisions needed before Phase 1

**D1 — Licence posture.** Permissive (RF-DETR/D-FINE + own tracker, repo Apache-2.0) or accept
AGPL-3.0 across the whole project to use Ultralytics? *Recommend permissive.* (§2)

**D2 — Detector family.** RF-DETR (packaged, maintained, SAHI-supported) or a YOLO-family model
(YOLOX, Apache-2.0) if the P2-head ablation is a research contribution you want? *Recommend RF-DETR
unless the P2 ablation matters to the dissertation.* (§3.3)

**D3 — Multi-class data.** Six of seven specified datasets are single-class. Either (a) add a
verified multi-class dataset such as AOD-4, (b) hand-annotate bird/aircraft tracks, or (c) narrow
the hypothesis to drone-vs-not-drone. **§4.3's confusion matrix is not deliverable without this.**
(§5.2)

**D4 — Hypothesis test design.** Add the matched appearance-classifier arm so the comparison is
controlled? *Recommend yes — without it there is no hypothesis test.* (§6.2)

**D5 — URL fetch.** Cut from v1 per §5.4's own escape hatch? *Recommend cut.* (§11.3)

### Open questions

**Q1 — OpenAI pricing.** `[UNKNOWN]`, egress-blocked. Needs checking from an unblocked network.
Model IDs *are* verified (§10).

**Q2 — GPU budget and host.** What is the actual monthly budget, and does the portfolio demo need a
persistent GPU or can visitor jobs run on CPU? (§11.2)

**Q3 — CUDA build variant** for torch 2.13.0. `[UNKNOWN]`, egress-blocked. Needed before the
Dockerfile. (§9.3)

**Q4 — DUT Anti-UAV split sizes and annotation format.** Not in the README; secondary numbers exist
but must not be cited. Confirm by downloading. (§5.1)

**Q5 — Drone-vs-Bird class field.** Are birds annotated as a class, or only present as distractors?
Determines whether D3 is solvable with data already specified. (§5.1, §5.2)

**Q6 — Licences for Det-Fly, MAV-VID, NPS-Drones, AOD-4, and UAVDetectionTrackingBenchmark.**
All `[UNKNOWN]`. (§5.1)

### Risks

**R1 — No GPU in this environment.** No real metrics until one is rented. (§12)

**R2 — Track scarcity may under-power the hypothesis test.** Low hundreds of drone tracks, unknown
and probably small number of bird tracks. *Mitigation: count usable tracks per class as the first
task of Phase 2, before writing converters. If bird tracks are under ~50, the framing must change.*
(§5.3)

**R3 — DUA lead time on Drone-vs-Bird.** A human must approve. *Mitigation: email
`wosdetc@googlegroups.com` on day one.* (§5.1)

**R4 — LRDDv2 export-control clearance** is unbounded-latency and may be refused. *Mitigation: treat
as optional, off the critical path.* (§5.1)

**R5 — Redistribution violation.** Drone-vs-Bird grants no redistribution rights. *Mitigation:
`.gitignore` for data/weights in commit 1, before any data is downloaded; DVC or git-lfs with an
external store; CI check that blocks image/video files in the repo.* (§5.1)

**R6 — Scope.** Ten phases spanning a research-grade evaluation harness *and* a hardened multi-tenant
web application is a large amount of work for one person. The security surface is driven almost
entirely by visitor mode. *Mitigation to consider: make the demo single-tenant behind one shared
password, which removes most of §5.1's registration, enumeration, lockout and per-record
authorisation work. The research contribution is unaffected.* Raising this now because scope is
cheapest to cut before it is built — **but the brief asks for multi-user, and it is your call.**

**R7 — Python 3.11 vs 3.12 split** between this container and the target image. *Mitigation: settle
on 3.12 and rebuild the dev container before Phase 1.* (§9.1)


---

## 14. Decision record

Decisions D1–D5 were put to the developer on 2026-08-25 and all five resolved to the recommended
option. Recorded here so future sessions inherit them; these belong in `CLAUDE.md` at Phase 1.

### 14.1 Resolved

| # | Decision | Resolution | Consequence |
|---|---|---|---|
| **D1** | Licence posture | **Permissive.** RF-DETR + own tracker; repo Apache-2.0 | No Ultralytics, no BoxMOT anywhere in the dependency tree. ByteTrack association and the constant-velocity Kalman filter are written in-project. CI should fail on an AGPL dependency appearing. |
| **D2** | Detector family | **RF-DETR** (Apache-2.0) | Behind a swappable interface so D-FINE can be benchmarked. Note: DETR-family has no P2 head in the YOLO sense — the §3.1 P2 ablation is **dropped** as a research contribution; input resolution and slice geometry become the equivalent knobs. |
| **D3** | Multi-class data | **Add a verified multi-class dataset** | Precondition **not met** — see §14.2. Do not treat this as closed. |
| **D4** | Hypothesis test | **Build both arms** | Phase 5 now delivers a matched pair: appearance arm (track crops → small CNN → temporal vote) and motion arm (motion features → GBT), on the same tracks, same pixel buckets, same metric, grouped splits. Deliverable is two curves against pixels-on-target with confidence intervals. |
| **D5** | URL fetch | **Cut from v1** | Upload-only. §5.4's SSRF surface, the `*.googlevideo.com` allowlist problem and the YouTube ToS exposure all disappear. `yt-dlp` comes out of the dependency list. Revisit deliberately later if ever. |

### 14.2 D3 is not actually closed — three problems

The chosen option was *"add a **verified** multi-class dataset"*. I could not complete the
verification, and while trying I found two substantive problems that are independent of licence.

**Problem 1 — the licence is unverifiable from this container.** Every route is egress-blocked:

```
data.mendeley.com    -> 000    api.datacite.org -> 000
pmc.ncbi.nlm.nih.gov -> 000    doi.org          -> 000
www.ncbi.nlm.nih.gov -> 000    api.crossref.org -> 000
www.sciencedirect.com-> 000    arxiv.org        -> 000
opus.lib.uts.edu.au  -> 000
```
`[VERIFIED: curl -o /dev/null -w '%{http_code}' per host, 2026-08-25]`

AOD-4's licence is therefore `[UNKNOWN]`. I am not assuming CC BY because Mendeley Data commonly
uses it. **Must be checked from an unblocked network before any download.**

**Problem 2 — AOD-4 appears to be built partly from Anti-UAV, which contaminates the
cross-dataset test.** AOD-4 is reported to be compiled from YouTube-8M, Anti-UAV, and a Roboflow
dataset, with videos converted to frames. *(secondary — search synthesis of the Data in Brief
paper; not verified against the paper itself, which is blocked here.)*

If true, two things follow:
- Training on AOD-4 and testing on Anti-UAV would leak: some test frames may have been in
  training. That silently inflates the §4.3 cross-dataset generalisation number — precisely the
  kind of fabricated-looking result §1.3 exists to prevent.
- Frames sourced from YouTube cannot be relicensed by a depositor who does not own them, so a
  permissive label on the Mendeley record may not cover the underlying imagery. Relevant because
  Tayr is a public repo.

**Mitigation if AOD-4 is used: treat Anti-UAV as contaminated with respect to AOD-4 and never use
that pair for the cross-dataset test.** Establish overlap by frame hashing before training.

**Problem 3 — and this is the one that matters most — AOD-4 probably does not fix the actual
research problem.** AOD-4 is an *image* dataset. Per §5.3, the motion classifier's training unit is
a **track**, not a frame. Bird bounding boxes in shuffled frames give the *detector* a bird class,
but yield **zero bird tracks** unless the frames are contiguous and identifiable per source video.

So D3 as resolved likely fixes:
- ✅ the detector's multi-class problem, and the §4.3 detection confusion matrix
- ❌ **not** the §4.1-stage-6 track classifier's bird problem, which is the hypothesis itself

**Recommended next step, before any download:** check whether AOD-4 frames carry source-video
identifiers and are contiguous. If they do not, the D4 hypothesis test still has no bird tracks,
and the realistic options narrow to hand-annotating bird tracks from video, or narrowing the
hypothesis. **This should be settled in Phase 2 task 1 (the track census, R2) — not later.**

### 14.3 Dependency changes from these decisions

Removed from §9.2: `yt-dlp` (D5). Never added: `ultralytics`, `boxmot` (D1).
Added at Phase 5: a small CNN for the appearance arm — `torchvision` (BSD) already covers this,
no new dependency (D4).
