# Temporal-Scale Modeling and Audio-Text Alignment for Audio Moment Retrieval with Zero-Training Saliency Reuse

Official code for the DCASE 2026 Workshop paper by **Seungdeok Choi** and **Yong-Hwa Park** (KAIST),
submitted to [DCASE 2026 Challenge Task 6: Audio Moment Retrieval from Long Audio](https://dcase.community/challenge2026/task-audio-moment-retrieval-from-long-audio)
(team `Choi_KAIST`, joint 6th of 21 teams).

Given a long recording and a free-form text query, the system returns the start and end times of the
matching moments. It is a compact QD-DETR detector on frozen M2D-CLAP features, built on two principles:

- **Temporal-scale and context-aware localization**: coarse auxiliary supervision, denoising query
  training, a single-layer bidirectional Mamba audio-memory encoder, and a bounded localized cascade.
- **Multi-granularity audio-text alignment**: span-level re-ranking, boundary-level contrastive, and
  frame-level saliency objectives.

At inference, **zero-training saliency reuse** turns the trained frame-saliency head into an extra
proposal stream, with no new parameters, objectives or training.

## Results

CASTELLA development-testing (1,347 queries) and the official DCASE 2026 blind evaluation set.

| System | R1@0.5 | **R1@0.7** | mAP | mAP@0.5 | mAP@0.75 | Blind R1@0.5 | Blind **R1@0.7** |
|---|---|---|---|---|---|---|---|
| Official baseline (MS-CLAP) | – | 13.59 | – | – | – | – | 13.56 |
| **System 3** (released checkpoint) | 48.40 | 31.03 | 25.82 | 43.33 | 24.94 | 55.93 | **41.24** |
| System 4 (2 seeds + saliency reuse) | 50.41 | 33.70 | 29.16 | 44.77 | 28.47 | 54.80 | 40.68 |

System 3 is our best single model on the blind set. Running the commands below with
`checkpoints/system3_seed2023.pth` reproduces the System 3 development-testing row exactly and regenerates
the System 3 blind-set predictions query for query.

## Repository layout

```
configs/
  system3_pretrain_clotho_moment.yml   # stage 1: Clotho-Moment pretraining
  system3.yml                          # stage 2: CASTELLA fine-tuning (= System 3)
checkpoints/
  system3_seed2023.pth                 # submitted System 3 model (weights only, 7.3 M parameters)
data/                                  # CASTELLA / Clotho-Moment annotations, blind-set query list
src/
  qd_detr.py                # detector, losses (SetCriterion), cascade refinement
  qd_detr_transformer.py    # T2A encoder, audio-memory encoder, conditional DETR decoder
  mamba_encoder.py          # bidirectional Mamba encoder
  dataset.py                # feature loading, span / saliency labels
  train.py                  # training (pretraining and fine-tuning)
  evaluate.py               # evaluation on development-validation / -testing
  create_submission.py      # blind-set submission (single model)
  infer_saliency_reuse.py   # zero-training saliency reuse + score-level ensembling (System 4)
  standalone_eval/          # official DCASE 2026 Task 6 metrics
tools/
  extract_m2d_features.py   # M2D-CLAP audio / text feature extraction
```

## Setup

Python 3.9, CUDA 11.8.

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
# Bi-Mamba CUDA kernels (recommended; a pure-PyTorch fallback with identical outputs is used otherwise)
pip install --no-build-isolation causal-conv1d==1.2.2.post1 mamba-ssm==2.2.2
```

## Features

The model uses frozen **M2D-CLAP 2025** features. Each 1 s audio window gives one 768-d embedding,
and the BERT text encoder gives 768-d token-level query embeddings. These features are not distributed
by the challenge, so they have to be extracted from the audio.

1. Clone [nttcslab/m2d](https://github.com/nttcslab/m2d) and download the
   [M2D-CLAP_2025 weights](https://github.com/nttcslab/m2d/releases/download/v0.5.0/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025.zip),
   then `pip install librosa timm einops nnAudio transformers`.
2. Get the audio of [CASTELLA](https://zenodo.org/records/17412176),
   [Clotho-Moment](https://zenodo.org/records/17129257) and the DCASE 2026 evaluation set.
3. Extract features into `features/` (the paths the configs expect):

```bash
M2D="--m2d_dir m2d --ckpt m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth"

python tools/extract_m2d_features.py $M2D --audio_dir /path/to/CASTELLA/audio \
    --jsonl data/castella_{train,val,test}_release.jsonl \
    --out_audio features/castella/m2d_clap --out_text features/castella/m2d_clap_text

python tools/extract_m2d_features.py $M2D --tar_dirs /path/to/Clotho-Moment/{train,valid,test} \
    --jsonl data/clotho_moment_{train,val,test}_release.jsonl \
    --out_audio features/clotho-moment/m2d_clap --out_text features/clotho-moment/m2d_clap_text

python tools/extract_m2d_features.py $M2D --audio_dir /path/to/dcase2026_evaluation_audio \
    --jsonl data/dcase2026_evaluation.jsonl \
    --out_audio features/dcase2026_eval/m2d_clap --out_text features/dcase2026_eval/m2d_clap_text
```

## Evaluate the released System 3 model

```bash
# development-testing (use -s val for development-validation)
python src/evaluate.py -c configs/system3.yml -m checkpoints/system3_seed2023.pth -s test

# blind-set submission -> results/system3/private_submission.jsonl
python src/create_submission.py -c configs/system3.yml -m checkpoints/system3_seed2023.pth
```

## Train

Both stages run 200 epochs with batch size 32 on a single RTX 3090. The checkpoint with the best
development-validation R1@0.7 is kept.

```bash
# Stage 1: Clotho-Moment pretraining
python src/train.py -c configs/system3_pretrain_clotho_moment.yml

# Stage 2: CASTELLA fine-tuning
python src/train.py -c configs/system3.yml \
    -r results/system3_pretrain_clotho_moment/best_checkpoint.pth
```

Run-to-run variation is substantial (System 3: R1@0.7 = 28.83 ± 2.16 over 6 seeds). Change `seed`
in the configs to train more seeds.

Systems 1 and 2 of the paper are the same pipeline with fewer components. Disable the corresponding
flags in **both** configs:

| | `use_coarse_aux` | `use_span_rerank` | `use_mamba_backbone` | `use_boundary_contrast` | `use_dn` | `use_cascade_refine` |
|---|---|---|---|---|---|---|
| System 1 | ✓ | ✓ | | | | |
| System 2 | ✓ | ✓ | ✓ | ✓ | ✓ | |
| System 3 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Zero-training saliency reuse and System 4

`infer_saliency_reuse.py` pools the cascade-refined decoder proposals of one or more checkpoints with
their saliency-derived proposals (threshold τ_sal = 0.5, at most 10 per checkpoint). It ranks the pool
by score and returns the top 10.

```bash
# System 3 + saliency reuse (development-testing R1@0.7 = 32.37)
python src/infer_saliency_reuse.py -c configs/system3.yml --ckpts checkpoints/system3_seed2023.pth -s test

# System 4: two independently trained seeds + saliency reuse
python src/infer_saliency_reuse.py -c configs/system3.yml \
    --ckpts checkpoints/system3_seed2023.pth results/system3_seed42/best_checkpoint.pth -s test   # or -s blind
```

Only the seed-2023 checkpoint is released. The paper's System 4 adds a second model fine-tuned with
`seed: 42`. `--no_saliency` evaluates the decoder proposals alone.

## Citation

```bibtex
@inproceedings{choi2026b,
    author = "Choi, Seungdeok and Park, Yong-Hwa",
    title = "Temporal-Scale Modeling and Audio-Text Alignment for Audio Moment Retrieval with Zero-Training Saliency Reuse",
    booktitle = "Proceedings of the 11th Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE 2026)",
    address = "Boston, USA",
    month = "October",
    year = "2026",
    pages = "236--240",
    abstract = "Audio moment retrieval (AMR) grounds free-form text queries as temporal spans in audio recordings. Target events range from short transients to tens-of-seconds scenes, so the tolerance of the tight-IoU metric varies substantially across queries. We address this with two design principles: modeling across temporal scales and aligning audio and text at multiple granularities. Our detector builds on QD-DETR over a single frozen M2D-CLAP encoder. For temporal scale, a coarse auxiliary branch supervises the shared encoder at halved resolution, denoising query training restores perturbed spans, a single-layer bidirectional Mamba encoder carries minute-long context, and a bounded cascade decoder re-attends to localized regions. For alignment, span re-ranking and boundary-contrastive objectives tie span and boundary regions to the query text, while a frame-level saliency objective aligns individual frames. At inference, zero-training saliency reuse converts this trained saliency head into a parallel proposal stream for proposal generation. Submitted to DCASE 2026 Task 6 as four cumulative variants, the system ranked joint sixth of 21 teams with 8.0 M trainable parameters per model. Paired per-query significance tests with false-discovery-rate control show that each stage improves at least one metric significantly, while several single-stage R1@0.7 differences remain within seed variability. Stratified analysis further shows that the largest gains occur for short, many-moment queries, where proposal coverage and boundary precision become tightly coupled at one-second temporal resolution.",
    isbn = "979-8-234-25461-0",
    doi = "10.5281/zenodo.22954971"
}
```

## Acknowledgements

This code builds on the [DCASE 2026 Task 6 baseline](https://github.com/awkrail/dcase2026_task6_baseline)
and [Lighthouse](https://github.com/line/lighthouse) (QD-DETR, Moment-DETR). It also uses
[M2D-CLAP](https://github.com/nttcslab/m2d) and [Mamba](https://github.com/state-spaces/mamba).
Files inherited from these projects keep their original license headers. This work was supported by
the Korea Institute for Advancement of Technology (KIAT) grant funded by the Korea Government (MOTIE)
(RS-2025-02263945, HRD Program for Industrial Innovation).

## License

MIT License (see [LICENSE](LICENSE)). Portions derived from the DCASE 2026 Task 6 baseline (MIT, welix); files adapted from Lighthouse and Moment-DETR keep their original license headers.
