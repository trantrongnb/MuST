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


## Licence

MIT, see [LICENSE](LICENSE). `videotransforms/` is vendored from
[torch_videovision](https://github.com/hassony2/torch_videovision) (MIT).
