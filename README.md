# MuST — Multi-granularity Sub-Text Queries for Few-Shot Action Recognition

A deliberately small CLIP-based few-shot action recognition model. The sub-text
catalogs and the episodic splits are in this repository; only the CLIP ViT-B/16
weights and the video frames are fetched separately.

The whole method is four steps:

1. an LLM decomposes each class into `N = 4` atomic sub-texts (offline, once);
2. **contiguous sub-texts are concatenated into `M = 10` prompts at four temporal
   granularities** — the proposed component, zero trainable parameters;
3. one cross-attention block lets each prompt read the `T = 8` CLIP frame
   features, **biased toward the frames its span actually covers** — a temporal
   prior the pyramid hands over for free, costing `N + 1 = 5` parameters;
4. mean cosine similarity against the support prototypes, trained with plain
   cross-entropy.

See [METHOD.md](METHOD.md) for the equations and the ablation protocol.

```text
Archery -> ["nock arrow", "draw bow", "aim bow", "release arrow"]

  length 1   nock arrow.                                              (x4)
  length 2   nock arrow. Then draw bow.                               (x3)
  length 3   nock arrow. Then draw bow. Then aim bow.                 (x2)
  length 4   nock arrow. Then draw bow. Then aim bow. Then release arrow.  (x1)
                                                                  M = 10 queries

Each query knows which stretch of the clip it describes, so attention is biased
toward it. The unpenalised region widens with span length, and the widest query
covers everything and stays unbiased:

  span            f1     f2     f3     f4     f5     f6     f7     f8
  "nock arrow"  -0.00  -0.00  -0.03  -0.28  -0.78  -1.53  -2.53  -3.78
  ...
  full span     -0.00  -0.00  -0.00  -0.00  -0.00  -0.00  -0.00  -0.00
```

## Layout

```text
MuST/
|-- train.py / eval.py              # entry points
|-- must/
|   |-- subtext.py                  # Steps 1-2: composition + span geometry
|   |-- attention.py                # Step 3: span-aligned cross-attention
|   |-- model.py                    # Step 4: prototype matching, the MuST module
|   |-- backbone.py                 # frozen CLIP ViT-B/16 wrapper
|   |-- lora.py                     # low-rank adaptation of CLIP (default on)
|   |-- temporal.py                 # motion-aware frame features (default on)
|   |-- dataset.py                  # episodic frame loader
|   |-- options.py                  # CLI arguments
|   `-- utils.py
|-- videotransforms/                # clip-level image transforms (vendored, MIT)
|-- data/sub_texts/                 # HMDB51, UCF101, Kinetics, SSv2-small catalogs
|-- splits/{hmdb,ucf,kinetics,ssv2_small}/
|-- pretrained/clip-vit-base-patch16/   # CLIP weights, fetched (not committed)
|-- scripts/
`-- tests/
```

Importing `must` pulls in only the sub-text machinery, so the text-only tools do
not pay for loading torch and transformers. The model is imported explicitly
with `from must.model import MuST`.

## Setup

```bash
git clone <this-repo> MuST && cd MuST
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 scripts/download_clip.py          # ~600 MB into pretrained/
```

Every script in `scripts/` sources [`scripts/env.sh`](scripts/env.sh), which
takes two overridable variables and nothing machine-specific:

| Variable | Default | Meaning |
|---|---|---|
| `PYTHON` | `python3` on `PATH` | interpreter to run |
| `DATA_ROOT` | `./data/datasets` | parent directory of the frame datasets |

```bash
PYTHON=~/miniconda3/envs/must/bin/python DATA_ROOT=/mnt/datasets \
  bash scripts/train_hmdb.sh 1
```

`env.sh` also points `CUDA_MPS_PIPE_DIRECTORY` at a writable directory, which
some CUDA installations require in order to initialise; set `MUST_SET_MPS_DIR=0`
to skip that.

### Datasets

Expected frame layout, with class names matching the keys of the sub-text
catalog:

```text
$DATA_ROOT/<dataset>/<class_name>/<video_id>/<frame>.jpg
```

The defaults are `hmdb_root`, `UCF101`, `kinetics_FSAR` and `ssv2_small_FSAR`
under `$DATA_ROOT`; override one at a time with `HMDB_ROOT`, `UCF_ROOT`,
`KINETICS_ROOT` or `SSV2_ROOT`. Running `train.py` directly instead of through a
script, set `MUST_DATA_ROOT` to move the default data root, or pass `--dataset`.

## Verify the install

```bash
python3 -m pytest tests -q                   # 34 tests
```

`tests/test_subtext.py` needs neither GPU nor CLIP; the other two build the model
and therefore load the CLIP checkpoint. The suite runs on CPU — prefix with
`CUDA_VISIBLE_DEVICES=""` if the GPU is busy.

```bash
python3 scripts/check_subtext_length.py \
    --subtext_path data/sub_texts/ucf101_class_subtexts.json
```

## Train

```bash
# usage: bash scripts/train_<dataset>.sh [shot] [granularity] [span_prior] [extra args]
bash scripts/train_hmdb.sh 1                      # 5-way 1-shot, pyramid + box prior
bash scripts/train_hmdb.sh 5
bash scripts/train_kinetics.sh 1
bash scripts/train_hmdb.sh 1 pyramid off          # ablate the prior away
bash scripts/train_hmdb.sh 1 atomic  box          # ablate the pyramid away
```

Anything after the third argument is forwarded verbatim to `train.py`:

```bash
bash scripts/train_hmdb.sh 1 pyramid box --lora_rank 8 --temporal_context transformer
```

Unless `--checkpoint_dir` says otherwise, checkpoints and logs go to

```text
work/<dataset>/<K>-shot/<granularity>_<prior>_<backbone>/
```

where `<backbone>` encodes the visual configuration (`lora8_transformer`,
`clip1`, `frozen`, ...). The backbone-adaptation ablation rows differ in neither
granularity nor prior, so without that segment they would overwrite each other.

## Evaluate

```bash
bash scripts/eval.sh work/hmdb/1-shot/pyramid_box_lora8_transformer/checkpoint_best.pt hmdb 1
```

`eval.py` reads the model-shaping arguments — granularity, span prior, LoRA,
temporal context — back out of the checkpoint's own stored args, so evaluation
cannot silently disagree with training. `eval.sh` accepts a trailing
`<granularity> <span_prior>` pair that *overrides* what the checkpoint says; it
is kept only for backwards compatibility, so leave it out unless you mean it.
Disable the behaviour with `--no-config_from_checkpoint`.

Each run appends a line to `eval_<way>way_<shot>shot.txt` next to the checkpoint.

## Ablations

```bash
bash scripts/run_ablation.sh hmdb 1                # the 2x2 grid plus the control
bash scripts/run_granularity_sweep.sh hmdb 1 off   # every granularity, prior off
```

Both scripts pass an explicit `--checkpoint_dir` per row (`ablation_*` and
`sweep_*` under `work/<dataset>/<K>-shot/`) and print a summary at the end.

| `--granularity` | M | Meaning |
|---|---|---|
| `global` | 1 | CLIP class prompt only, no sub-texts |
| `atomic` | 4 | Atomic sub-texts as independent queries |
| `full` | 1 | One concatenation of all four sub-texts |
| `atomic_full` | 5 | Finest + coarsest level only |
| `pairs` | 7 | Levels 1–2 |
| `pyramid` | 10 | **Proposed** — all four levels |

| `--span_prior` | Meaning |
|---|---|
| `off` | Plain cross-attention, no temporal bias |
| `box` | **Proposed** — bias each query toward the frames its span covers |
| `random` | Control: same widths, permuted centers |

| `--temporal_context` | Parameters | Meaning |
|---|---:|---|
| `transformer` | 364,289 | **default** — motion transformer over appearance + frame differences |
| `conv` | 1,537 | depthwise `k=3` control — how much needs a full transformer? |
| `off` | 0 | plain CLIP frame features |

### Switches that are not part of the paper

`--use_exclusive`, `--visual_adapter`, `--prepend_class_name` and
`--clip_train_last_layers` are exploratory switches kept for completeness. **All
are off by default**, so the default configuration is exactly the method
described in METHOD.md — no auxiliary loss, no second adapter, no global branch.
Also available: `--warmup_iterations`, `--scalar_lr_mult`.

## Backbone adaptation and motion features

Neither is a contribution of the paper — see [METHOD.md §3](METHOD.md), "What is
and is not claimed". Both are held fixed across every ablation row so they cannot
be confused with the two proposed components.

### LoRA instead of unfreezing blocks (default)

`--clip_train_last_layers 1` makes 7.48M parameters trainable — more than three
times the whole MuST head — to adapt one block. In the 5-way 1-shot regime the
binding constraint is overfitting, not capacity, so this is the wrong direction.
LoRA has the right *shape*: it reaches every block through a rank-`r`
bottleneck, so adaptation is broad and shallow rather than narrow and deep.

| Mode | Trainable on CLIP | Total trainable |
|---|---:|---:|
| `--lora_rank 8`, CLIP frozen (**default**) | 294,912 | 2,398,726 |
| `--clip_train_last_layers 1`, no LoRA | 7,482,624 | 9,586,438 |
| CLIP fully frozen, no LoRA | 0 | 2,103,814 |

Totals exclude the motion transformer; with it at its default the full model
trains 2.76M parameters. To recover the unfrozen-block behaviour for the
backbone-adaptation ablation:

```bash
bash scripts/train_hmdb.sh 1 pyramid box --lora_rank 0 --clip_train_last_layers 1
```

Two things will bite you:

- **LoRA needs `--lora_lr 1e-4`**, about two orders above the `2e-6` used for
  `--clip_learning_rate`. Reusing the fine-tuning rate is the most common way to
  make LoRA silently learn nothing.
- **`--clip_gradient_checkpointing` becomes mandatory.** LoRA makes every block
  trainable, so activations are retained for all twelve; without checkpointing
  this OOMs on a 32 GB card where `--clip_train_last_layers 1` fits.

`B` is zero-initialised, so a wrapped layer is bit-exact with the frozen original
at step 0 — which matters here because the `M` text queries are encoded once by
frozen CLIP into a buffer, and training must start from precisely the model that
buffer was built for.

### Motion-aware temporal context

CLIP encodes each frame independently, so its features carry appearance only and
cannot separate an action from its time-reversed version. The span prior decides
*where along the clip* each query looks; it cannot invent motion information the
features never had. `--temporal_context transformer` inserts a bottleneck
Transformer that reads the frame embeddings **together with their first-order
differences** and writes a zero-gated residual back into CLIP space.

### Read the diagnostics before believing any of it

Both branches are zero at initialisation by design, so training prints their
state on every progress line:

```text
Task [150/400], LR: 2.5e-05, Acc: 0.810, L_ce: 0.5339, motion: -0.0194, lora_b: 0.0010
```

`motion` is `tanh(g)` of the temporal context, `lora_b` is mean `|B|`. Both are
exactly `0` at initialisation. **A value still at zero after a few thousand
iterations means that branch never activated** and any result attributed to it is
void — almost always a learning-rate mistake. Zero-initialising a gate *and* its
branch produces a dead saddle with zero gradient on both, which is why
`TemporalContext` seeds its kernel with a central-difference operator and zeroes
only the gate.

## Inspect what the prior learned

```bash
python3 scripts/plot_span_prior.py \
    -pc work/hmdb/1-shot/pyramid_box_lora8_transformer/checkpoint_best.pt \
    --output fig_span_prior.png
```

Prints the learned gate per span length and the `[M, T]` bias matrix, and
optionally saves the two-panel figure. Always pass a trained checkpoint: without
`-pc` it shows the initialisation, where every gate is 0.5 by construction.

## Sub-text catalogs

`data/sub_texts/*.json` maps a class name to its `N` atomic sub-texts:

```json
{"Archery": ["nock arrow", "draw bow", "aim bow", "release arrow"]}
```

**Sub-texts must be short.** Concatenations have to fit CLIP's 77-token window;
when they overflow, the longest spans get truncated to the same prefix and the
coarse granularities silently disappear. Check any catalog before training with:

```bash
python3 scripts/check_subtext_length.py --subtext_path <catalog.json>
```

The script exits non-zero and names the offending prompts if any would be cut.
Short verb phrases of a few words each keep every span inside the window; long
descriptive sentences do not. Training also prints a warning at start-up when any
prompt would be truncated.

The catalogs shipped here were generated once per dataset from class names alone
with:

```text
Action: {class name}
Decompose this action into exactly 4 consecutive atomic steps.
Each step must be 2-5 words, lowercase, no punctuation, describing a single
visible movement. Return one step per line, no numbering.
```

The phases must come back **in temporal order** — the span prior reads a
sub-text's index as a temporal coordinate, so a shuffled catalog degrades it to
the `random` control.

## Licence

MIT, see [LICENSE](LICENSE). `videotransforms/` is vendored from
[torch_videovision](https://github.com/hassony2/torch_videovision) (MIT).
