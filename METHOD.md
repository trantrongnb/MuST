# MuST: Multi-granularity Sub-Text Queries for Few-Shot Action Recognition

Method specification. Everything below describes the reference configuration
`pyramid_box_lora8_transformer`, i.e. the run

```text
--granularity pyramid --span_prior box
--clip_train_last_layers 0 --lora_rank 8 --lora_targets qv
--temporal_context transformer
```

No experimental results are reported here.

---

## 1. Problem setting

An episode is a `W`-way `K`-shot task drawn from a class-disjoint split:

```text
W = 5        classes per episode
K = 1 or 5   labelled support videos per class
T = 8        frames sampled per video
D = 512      CLIP ViT-B/16 embedding width
N = 4        atomic sub-texts per class (read from the catalogue)
M = 10       text queries per class, = N(N+1)/2 for the pyramid
```

A query video must be assigned to one of the `W` candidate classes using only
the `W * K` labelled support videos of that episode. Training uses 6 query
videos per class per episode, evaluation uses 1.

## 2. Motivation

CLIP-based few-shot action recognition usually renders a class as one sentence,
`"a video of a person performing archery"`. That single vector is a snapshot: it
says what an action looks like, not how it unfolds. Recent work therefore asks
an LLM to decompose each class into `N` atomic phases:

```text
Archery -> ["nock arrow", "draw bow", "aim bow", "release arrow"]
```

This fixes the snapshot problem only halfway. A *set* of `N` phrases is
invariant to permutation, so it cannot separate *"draw bow then release arrow"*
from *"release arrow then draw bow"*, and nothing tells the query *"nock arrow"*
to read the beginning of the clip rather than its end. The usual remedy adds a
temporal module on the visual side, which costs parameters.

**MuST builds the temporal structure in language space instead.** Contiguous
sub-texts are concatenated back into one sentence and re-encoded by the frozen
CLIP text encoder. A longer concatenation describes a longer stretch of the
action, so a class becomes a *pyramid of text queries at several temporal
granularities* — obtained by string concatenation alone, at zero parameter cost.
Because each such query then has a known temporal extent, the same construction
also yields a temporal alignment prior for the cross-attention.

## 3. What is and is not claimed

| Component | Origin | Trainable parameters |
|---|---|---|
| Sub-text catalogue from an LLM | prior work (SFAR) | 0 |
| **Multi-granularity sub-text pyramid** | **contribution 1** | **0** |
| **Span-aligned cross-attention prior** | **contribution 2** | **N + 1 = 5** |
| Text-to-frame cross-attention block | standard | 2,103,808 |
| LoRA on the visual tower | Hu et al. 2021 | 294,912 |
| Motion-aware temporal context | adopted, see §7.2 | 364,289 |
| Prototype matching + cross-entropy | ProtoNet | 1 (logit scale) |

The two claimed contributions cost **5 parameters between them**. The other rows
are adopted machinery; §12 states how they must be held fixed or ablated so that
their effect is not credited to the pyramid.

---

## 4. Step 1 — Sub-text catalogue (adopted)

An LLM is prompted once per class, offline, with the class name alone and no
video, to decompose the action into `N = 4` consecutive atomic phases. The
result is a static JSON file:

```json
{"Archery": ["nock arrow", "draw bow", "aim bow", "release arrow"]}
```

The catalogue is produced once per dataset and takes no part in training. Two
requirements:

- **Temporal order.** The phases must be returned in the order they occur. §6
  depends on this; without it the interval assigned to a span no longer matches
  its content.
- **Brevity.** The concatenations must fit CLIP's 77-token context window. Long
  descriptive sentences overflow it, the length-3 and length-4 spans are then
  truncated to a shared prefix, and the coarse levels silently disappear. Verify
  with `scripts/check_subtext_length.py` before training.

## 5. Step 2 — Multi-granularity pyramid (contribution 1)

Let `S_c = [s_c1, ..., s_cN]` be the atomic sub-texts of class `c`. Keep every
contiguous span,

```text
Omega = { (i, j) : 1 <= i <= j <= N },      M = |Omega| = N(N+1)/2
```

For `N = 4` this gives `M = 10` spans on four nested levels:

```text
length 1   (1,1) (2,2) (3,3) (4,4)     4 single-phase queries
length 2   (1,2) (2,3) (3,4)           3 transition queries
length 3   (1,3) (2,4)                 2 phase-group queries
length 4   (1,4)                       1 whole-action query
```

Each span is rendered back into one sentence by the join operator

```text
Join(s_i, ..., s_j) = "s_i. Then s_{i+1}. Then ... Then s_j."
```

and encoded by the frozen CLIP text encoder:

```text
q_c,w = normalize( CLIP_text( Join(s_ci, ..., s_cj) ) )  in R^D,   w = (i,j)
Q_c   = [ q_c,w ]_{w in Omega}                           in R^{M x D}
```

`Q_c` is computed once at start-up and stored as a buffer — 51 x 10 x 512 =
261,120 values for HMDB51. It never receives a gradient, so this component adds
**zero trainable parameters**.

Two properties make the construction work:

- **Order sensitivity.** A text encoder consumes a token sequence, so
  `"draw bow. Then release arrow."` and `"release arrow. Then draw bow."` map to
  different embeddings. A set of atomic phrases cannot express that distinction.
- **Granularity coverage.** Short spans localise brief movements; the longest
  span behaves like a conventional global prompt. The `M` queries are matched
  **individually**, never averaged, so a video must agree with the class at every
  temporal scale rather than only in aggregate.

Raising `M` from 4 to 10 adds rows to a frozen buffer and changes no parameter
count. It multiplies only the attention queries, negligible next to encoding
`T = 8` frames through the ViT.

## 6. Step 3 — Span-aligned cross-attention (contribution 2)

A single multi-head cross-attention block (8 heads, dropout 0.1) lets each text
query read the frames. Text queries are the attention queries; frame features,
plus a fixed sinusoidal temporal encoding `PE`, are keys and values.

### 6.1 The information a plain block discards

Nothing in standard cross-attention stops the query `"nock arrow"` — the first
of four phases — from attending to the last frame, where the arrow has already
been released. At `K = 1` this is easy to get wrong and the model receives no
signal that it is wrong.

But the queries are not arbitrary sentences. The LLM was asked for `N`
*consecutive* phases, so sub-text `k` corresponds to the k-th slice of the
timeline, and the span `w = (i,j)` covers a known interval:

```text
mu_w = (i - 1 + j) / (2N)       centre of the span, in normalised time
u_w  = (j - i + 1) / (2N)       half-width of the span
p_t  = (t - 0.5) / T            position of frame t
```

The mapping needs no learning and no extra annotation. It does need **both** a
centre and a width per query, which only nested spans supply: a global prompt
gives neither, and a flat set of atomic sub-texts gives a centre with a width
shared by every query. **Contribution 2 is therefore enabled by contribution 1
rather than merely stacked on it.**

### 6.2 The prior

Frames outside a query's interval are penalised on the attention logits, flat
inside the interval and growing quadratically outside:

```text
b[w, t] = -gate_|w| * ( relu( |p_t - mu_w| - u_w ) / decay )^2

A_v,c   = MHA( Q = LN(Q_c), K = V = LN(F_v + PE),  attn_mask = b )
H_v,c   = Q_c + Dropout(A_v,c)
P_v,c   = H_v,c + FFN( LN(H_v,c) )                   in R^{M x D}
```

`gate` holds one learned scalar per span length (`N = 4` of them) and `decay` one
shared scalar, so the prior costs exactly `N + 1 = 5` parameters. Both are stored
raw and passed through `softplus` to stay positive, so `gate` approaches zero
asymptotically rather than reaching it; `decay` is clamped at `1e-4`. Given the
geometry, `b` is a constant `[M, T]` matrix broadcast over batch and heads, so
its compute cost is nil. It reaches `nn.MultiheadAttention` through `attn_mask`
as a float tensor, which PyTorch adds to the logits. The geometry itself is a
frozen buffer (`span_center`, `span_half_width`, `span_level`, 10 values each).

At `N = 4`, `T = 8`, `gate = 0.5`, `decay = 0.25` the matrix is

```text
span              f1     f2     f3     f4     f5     f6     f7     f8
length 1       -0.00  -0.00  -0.03  -0.28  -0.78  -1.53  -2.53  -3.78
length 1       -0.28  -0.03  -0.00  -0.00  -0.03  -0.28  -0.78  -1.53
length 1       -1.53  -0.78  -0.28  -0.03  -0.00  -0.00  -0.03  -0.28
length 1       -3.78  -2.53  -1.53  -0.78  -0.28  -0.03  -0.00  -0.00
length 2       -0.00  -0.00  -0.00  -0.00  -0.03  -0.28  -0.78  -1.53
length 2       -0.28  -0.03  -0.00  -0.00  -0.00  -0.00  -0.03  -0.28
length 2       -1.53  -0.78  -0.28  -0.03  -0.00  -0.00  -0.00  -0.00
length 3       -0.00  -0.00  -0.00  -0.00  -0.00  -0.00  -0.03  -0.28
length 3       -0.28  -0.03  -0.00  -0.00  -0.00  -0.00  -0.00  -0.00
length 4       -0.00  -0.00  -0.00  -0.00  -0.00  -0.00  -0.00  -0.00
```

### 6.3 Three properties, all consequences of the formula

1. **The widest query is unbiased by construction.** A span of length `N` has
   `mu_w = u_w = 0.5`, every frame lies inside it, the ReLU is identically zero
   and that row of `b` vanishes — the last row above. Not a special case in the
   code, and true for every `N`.
2. **The prior is rejectable.** As `gate -> 0` the bias disappears and the block
   reduces to standard cross-attention, so the model can discard the prior where
   it is wrong. The learned gates are therefore themselves a reportable quantity.
3. **The penalty is soft.** "Phase `k` occupies the k-th slice" is an
   approximation — phases differ in duration, clips carry lead-in and lead-out,
   and the LLM does not guarantee strictly sequential steps. A quadratic penalty
   lets attention degrade gradually instead of forbidding it outright.

The third point is why the ablation must include a control that keeps every span
width but permutes the centres (`--span_prior random`). If the aligned prior
cannot beat that control, the gain is generic regularisation, not temporal
alignment.

## 7. Visual branch (adopted components)

The frame path is `CLIP -> temporal context -> L2-normalise`.

### 7.1 Frozen CLIP with LoRA

CLIP ViT-B/16 is **fully frozen** (`--clip_train_last_layers 0`). Adaptation is a
low-rank update on the query and value projections of all twelve visual blocks:

```text
y = W_0 x + (alpha / r) * B A Dropout(x),      B <- 0,  A ~ Kaiming
r = 8,  alpha = 16,  dropout = 0.05           294,912 parameters
```

LoRA is preferred over unfreezing whole blocks because the binding constraint in
this regime is **overfitting, not capacity**: a rank-`r` bottleneck reaches every
layer with 295K parameters, whereas making one block fully trainable costs 7.48M.
`B = 0` makes each wrapped layer bit-exact with the frozen original at step 0,
which the frozen text-query buffer requires.

`--lora_lr` runs at `1e-4`, about two orders above `--clip_learning_rate`. Too
low and the LoRA branch never moves, silently reverting the model to fully
frozen; the `lora_b` diagnostic exists to catch that.

### 7.2 Motion-aware temporal context

CLIP encodes each frame independently, so its features carry appearance only and
cannot separate an action from its time-reversed version. The span prior decides
*where along the clip* each query looks; it cannot invent motion information the
features never had. A bottleneck Transformer therefore reads the frame
embeddings together with their first-order differences and writes a gated
residual back:

```text
z_t   = W_down LN(u_t)                    d_t = z_t - z_{t-1},  d_1 = 0
z~    = Enc( z + phi(d) + PE )
h_t   = u_t + tanh(g) * W_up z~_t
```

`d = 128`, 4 heads, 1 layer, dropout 0.2, pre-norm — 364,289 parameters. The
difference stream supplies motion directly as a token feature instead of forcing
attention to infer it from appearance, and temporal self-attention relates
distant frames where a `k = 3` convolution sees only its neighbours.

This is **orthogonal to the span prior**: the prior fixes attention, this fixes
the features attention reads.

### 7.3 Zero-gated initialisation

Every residual branch is gated so that the frame features at initialisation are
*exactly* raw CLIP:

| Branch | Zero-initialised | State at iteration 0 |
|---|---|---|
| Motion transformer | gate `g` only (`W_up` random) | `h_t = u_t`, gradient on `g` non-zero |
| Temporal conv (ablation) | gate only; kernel seeded as central difference | `h_t = u_t`, gradient on `g` non-zero |
| LoRA | `B` | wrapped layer bit-exact with the frozen original |

This is not cosmetic. The `M` text queries are encoded once by frozen CLIP into a
buffer, so perturbing the visual features at step 0 would start the frozen
queries matched against features CLIP never produced.

> **Never zero-initialise a gate and its branch together.** Both gradients are
> then identically zero — a dead saddle the module never leaves. Measured, not
> hypothesised: in an earlier version of this model the motion gate sat at
> exactly `0.0000` for
> 8000 iterations on SSv2 and HMDB, i.e. the model had no motion cue at all.

## 8. Step 4 — Prototype matching

Support patterns of a class are averaged over its `K` shots:

```text
R_c = (1/K) * sum_k P_{x^s_{c,k}, c}      in R^{M x D}
```

Each query video is read once per candidate class, and the class score is the
mean cosine similarity across the `M` granularities:

```text
S(q, c) = tau * (1/M) * sum_{w in Omega} cos( P_{q,c}[w], R_c[w] )
```

`tau` is a learned logit scale, `exp`-parameterised and clamped at 100.
Averaging over granularities rather than over one fused vector requires a video
to agree with the class at every temporal scale. Training minimises plain
episodic cross-entropy:

```text
L = CE( S(q, .), y_q )
```

There is no support adaptation, no auxiliary loss and no separate global branch.
The adopted components of §7 are, however, real machinery: they are held fixed
across every configuration compared in §12, so differences between rows isolate
the two proposed components, but they are not absent from the model.

## 9. Tensor flow

```text
LLM (offline)          CLIP text (frozen)       CLIP visual (frozen + LoRA r=8)
     |                        |                            |
  N=4 atomic             M=10 prompts                  T=8 frames
  sub-texts  --[Step 2]-->  per class  ------>          u [T,D]
                                |                          |
                            Q_c [M,D]                 [§7.2] motion transformer
                                |                          |
                                |                     L2-normalise -> F_v [T,D]
                                |                          |
                                +----> [Step 3] span-aligned cross-attention
                                              (text Q, frames KV, bias b)
                                                    |
                                             P_v,c [M,D]
                                                    |
                              support: mean over K shots -> R_c [M,D]
                              query:   P_q,c [M,D]
                                                    |
                                [Step 4] mean cosine over M -> S(q,c)
                                                    |
                                           cross-entropy loss
```

## 10. Parameter budget

Counted from the reference checkpoint (HMDB51, `N = 4`, `M = 10`):

| Component | Trainable |
|---|---:|
| Cross-attention block (MHA + FFN + LayerNorms) | 2,103,808 |
| Span prior (4 gates + 1 shared decay) | 5 |
| Motion transformer (`d=128`, 4 heads, 1 layer) | 364,289 |
| LoRA `r=8` on q,v of 12 visual blocks | 294,912 |
| Logit scale `tau` | 1 |
| **Total** | **2,763,015** |

Frozen, not counted: the CLIP towers, the text-query buffer `Q_c` (261,120
values for HMDB51) and the span geometry (30 values).

For comparison, `--clip_train_last_layers 1` instead of LoRA would cost
7,482,624 parameters for a single block.

## 11. Training configuration

```text
Optimiser        AdamW, weight decay 5e-4
Learning rates   5e-5   MuST head
                 5e-4   scalars and 1-D tensors (x10, no weight decay)
                 1e-4   LoRA
Schedule         1,000 linear warm-up iterations, then step decay by
                 0.5 / 0.1 / 0.01 at 30% / 50% / 70% of the run
Iterations       10,000, gradient accumulation over 2 episodes
Frames           T = 8, one per equal segment: random during training,
                 segment centre at test time; 224x224
Augmentation     train  Resize(256), RandomHorizontalFlip, RandomCrop(224),
                        ColorJitter(0.4, 0.4, 0.4, 0.1)
                 test   Resize(256), CenterCrop(224)
Precision        mixed (AMP), gradient checkpointing on the visual tower
Model selection  best of 1,000 validation episodes, evaluated every 1,000
                 iterations on the VALIDATION split
Seed             1234
Hardware         1x RTX 5090
```

`checkpoint_best.pt` is selected on the validation split, so the test split is
untouched until `eval.py` and the reported accuracy is genuinely held out. Much
of the few-shot action recognition literature instead selects on test;
`--val_split test` reproduces that protocol, but then the number is a
model-selection score and must not be described as held out. Those runs are kept
on a separate `_seltest` path and the choice is recorded in the log, so the two
protocols can never be confused for one another.

The warm-up is not incidental. LoRA and the zero-gated branches start from an
exactly-frozen model (§7.3), and a full-rate first step can knock the frame
features away from the raw CLIP features that the frozen text-query buffer was
built against.

The seed also fixes the permutation used by `--span_prior random`, so every
ablation row must share it.

## 12. Ablation protocol

All rows share the same split, episodes, frame sampling, way, shot, seed,
schedule and sub-text catalogue. Contribution 1 is ablated on `--granularity`:

| `--granularity` | M | What it isolates |
|---|---|---|
| `global` | 1 | CLIP class prompt only, no sub-texts |
| `atomic` | 4 | Atomic sub-texts, no concatenation |
| `full` | 1 | One concatenation of all `N` — "why not just one long sentence?" |
| `atomic_full` | 5 | Finest + coarsest, no intermediate levels |
| `pairs` | 7 | Levels 1–2 only |
| `pyramid` | 10 | **Proposed** — all levels |

`full` is the row reviewers ask for first: it shows the gain comes from having
*several* granularities, not merely from a longer prompt. `atomic_full` versus
`pyramid` shows whether the intermediate levels earn their place.

Contribution 2 is ablated on the orthogonal axis `--span_prior`:

| `--span_prior` | Meaning |
|---|---|
| `off` | Plain cross-attention, no temporal bias |
| `box` | **Proposed** — bias each query toward the frames its span covers |
| `random` | Control: identical widths, permuted centres. Separates temporal alignment from generic attention regularisation |

The main table is the 2x2 grid `{atomic, pyramid} x {off, box}` plus the
`pyramid/random` control. Four readings follow from it:

```text
pyramid/off  - atomic/off      contribution of the pyramid alone
pyramid/box  - pyramid/off     contribution of the prior alone
pyramid/box  - pyramid/random  evidence the gain is alignment, not regularisation
(atomic/box - atomic/off) < (pyramid/box - pyramid/off)
                               evidence the two contributions interlock
```

The adopted components of §7 need their own axes, and **must** be ablated
separately or reviewers will attribute their effect to the pyramid:

| Axis | Reference | Ablation | Question answered |
|---|---|---|---|
| `--temporal_context` | `transformer` | `conv`, `off` | how much comes from motion features rather than from the pyramid? |
| Backbone adaptation | `--lora_rank 8`, CLIP frozen | `--clip_train_last_layers 1/4` | is the constraint overfitting rather than capacity? |

The temporal context is a large *independent* contributor. Introducing it in the
same run that introduces the pyramid confounds the two; a run pairing the motion
transformer with the `atomic` baseline is needed before either can be credited.

Two further switches, both off in the reference configuration:

- `--prepend_class_name` prefixes every composed prompt with the readable class
  name.
- `--use_exclusive` builds a second, *exclusive* pattern per query,
  `P^- = FFN_refine(Q_c - A_v,c)`, and **subtracts** a cross term from the score:
  `S <- S - lambda * 0.5 * [ cos(P^-_q, R^+_c) + cos(P^+_q, R^-_c) ]` with
  `lambda = --exclusive_weight` (default 0.25). A bridge to TEAM-style
  discriminative matching.

## 13. Relation to prior work

- **TEAM** learns `M` generic pattern tokens shared by every class. MuST replaces
  them with class-specific text queries: semantically grounded, and requiring no
  learning.
- **SFAR** uses the `N` atomic sub-texts directly as independent queries — the
  `atomic` row of the ablation. MuST adds the concatenation pyramid on top and
  derives the prior from it.
- **CLIP-FSAR** represents a class by one global prompt — the `global` row,
  `M = 1`.
- **TRN / TPN** build multi-scale temporal representations on the *visual* branch
  with extra convolutions and parameters. MuST obtains the same multi-scale
  effect on the *text* branch by string concatenation, at zero parameter cost.
- **CLIP prompt ensembling** averages many paraphrases of one class description
  into a single vector. The `M` prompts here are not paraphrases: they are nested
  temporal spans with different content, matched individually rather than
  averaged.

## 14. Generality in `N`

`N` is read from the catalogue; nothing hardcodes `N = 4`. The geometry uses
normalised time, so it stays valid when `N` does not divide `T`, and the widest
span stays unbiased for every `N`. Measured coverage of the atomic spans at
`T = 8`:

```text
N=3  M= 6  atomic spans cover [3,2,3] frames
N=4  M=10  atomic spans cover [2,2,2,2]      <- tiles T exactly
N=5  M=15  atomic spans cover [2,1,2,1,2]
N=6  M=21  atomic spans cover [1,2,1,1,2,1]
```

`N = 4` with `T = 8` tiles the clip exactly, which is the technical reason for
the default. From `N = 5` some atomic spans collapse onto a single frame and the
quadratic penalty becomes effectively a hard mask at that level; raise `T` to 16
instead. State the constraint as **`N <= T/2`**. Note that `gate` has shape `[N]`,
so checkpoints do not transfer across different `N`.
