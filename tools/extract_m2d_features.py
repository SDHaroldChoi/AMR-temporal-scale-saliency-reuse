"""Extract frozen M2D-CLAP 2025 audio and text features.

Audio: 16 kHz mono, non-overlapping 1 s windows -> encode_clap_audio -> (T, 768)
       saved as <out_audio>/<vid>.npz (key "features").
Text:  token-level hidden states of the M2D-CLAP BERT text encoder -> (L, 768)
       saved as <out_text>/qid<qid>.npz (key "last_hidden_state").

Requires the M2D repository (https://github.com/nttcslab/m2d) and the
M2D-CLAP_2025 weights (m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025).

    # CASTELLA (directory of <vid>.wav)
    python tools/extract_m2d_features.py --audio_dir /path/to/CASTELLA/audio \
        --jsonl data/castella_{train,val,test}_release.jsonl \
        --out_audio features/castella/m2d_clap --out_text features/castella/m2d_clap_text

    # Clotho-Moment (WebDataset tar archives with <key>.wav + <key>.json)
    python tools/extract_m2d_features.py --tar_dirs /path/to/Clotho-Moment/{train,valid,test} \
        --jsonl data/clotho_moment_{train,val,test}_release.jsonl \
        --out_audio features/clotho-moment/m2d_clap --out_text features/clotho-moment/m2d_clap_text

    # DCASE 2026 blind evaluation set
    python tools/extract_m2d_features.py --audio_dir /path/to/dcase2026_evaluation_audio \
        --jsonl data/dcase2026_evaluation.jsonl \
        --out_audio features/dcase2026_eval/m2d_clap --out_text features/dcase2026_eval/m2d_clap_text

All commands also take --m2d_dir (M2D repo) and --ckpt (checkpoint-30.pth).
"""
import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm

SAMPLE_RATE = 16000


def get_m2d_model(m2d_dir, ckpt_path, device):
    sys.path.insert(0, str(m2d_dir))
    sys.path.insert(0, str(Path(m2d_dir) / 'examples'))
    from portable_m2d import PortableM2D
    model = PortableM2D(ckpt_path, flat_features=True)
    return model.to(device).eval()


def sliding_windows(audio_source, win_sec=1.0, hop_sec=1.0):
    """Load audio (path or file-like) and cut it into (T, win_len) windows."""
    audio, _ = librosa.load(audio_source, sr=SAMPLE_RATE, mono=True)
    win_len = int(win_sec * SAMPLE_RATE)
    hop_len = int(hop_sec * SAMPLE_RATE)
    n_frames = max(1, (len(audio) - win_len) // hop_len + 1)
    frames = np.stack([audio[i * hop_len: i * hop_len + win_len] for i in range(n_frames)])
    return torch.from_numpy(frames)


@torch.no_grad()
def encode_windows(model, frames, device, batch_size=64):
    frames = frames.to(device)
    feats = [model.encode_clap_audio(frames[i: i + batch_size]).cpu().float().numpy()
             for i in range(0, len(frames), batch_size)]
    return np.concatenate(feats, axis=0)  # (T, 768)


def extract_audio_dir(audio_dir, out_dir, model, device, batch_size=64):
    """<audio_dir>/<vid>.wav -> <out_dir>/<vid>.npz"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for wav_path in tqdm(sorted(Path(audio_dir).glob('*.wav')), desc='audio'):
        out_path = out_dir / f'{wav_path.stem}.npz'
        if out_path.exists():
            continue
        try:
            frames = sliding_windows(str(wav_path))
        except Exception as e:
            print(f'  skip {wav_path.name}: {e}')
            continue
        np.savez(out_path, features=encode_windows(model, frames, device, batch_size))


def clotho_moment_vid(bg_path):
    """'.tmp/train/bg/Amsterdam/0.0_60.0.wav' -> 'Amsterdam_0.0_60.0' (the jsonl vid)."""
    parts = Path(bg_path).parts
    return f'{parts[-2]}_{Path(parts[-1]).stem}'


def extract_audio_tars(tar_dirs, out_dir, model, device, batch_size=64):
    """Clotho-Moment WebDataset tars: <key>.wav (mixture) + <key>.json (metadata)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for tar_dir in tar_dirs:
        for tar_path in tqdm(sorted(Path(tar_dir).glob('*.tar')), desc=f'audio {Path(tar_dir).name}'):
            with tarfile.open(tar_path, 'r') as tf:
                members = {m.name: m for m in tf.getmembers()}
                for key in sorted({Path(name).stem for name in members}):
                    wav_name, json_name = f'{key}.wav', f'{key}.json'
                    if wav_name not in members or json_name not in members:
                        continue
                    try:
                        meta = json.loads(tf.extractfile(members[json_name]).read())
                        vid = clotho_moment_vid(meta['bg']['path'])
                    except Exception as e:
                        print(f'  skip {key}: {e}')
                        continue
                    out_path = out_dir / f'{vid}.npz'
                    if out_path.exists():
                        continue
                    try:
                        frames = sliding_windows(io.BytesIO(tf.extractfile(members[wav_name]).read()))
                    except Exception as e:
                        print(f'  skip {key} audio: {e}')
                        continue
                    np.savez(out_path, features=encode_windows(model, frames, device, batch_size))


@torch.no_grad()
def extract_text(jsonl_paths, out_dir, model, device, max_length=32):
    """Token-level text features (CLS pooling bypassed), one file per qid."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not hasattr(model, 'text_encoder'):
        model.get_clap_text_encoder()
    text_enc = model.text_encoder  # BertXEncoder wrapping bert-base-uncased

    seen_qids = set()
    for jsonl_path in jsonl_paths:
        with open(jsonl_path) as f:
            data = [json.loads(line) for line in f]
        for item in tqdm(data, desc=f'text {Path(jsonl_path).name}'):
            qid = item['qid']
            if qid in seen_qids:
                continue
            seen_qids.add(qid)
            out_path = out_dir / f'qid{qid}.npz'
            if out_path.exists():
                continue
            inp = text_enc.tokenizer([item['query']], padding='longest', truncation=True,
                                     max_length=max_length, return_tensors='pt').to(device)
            out = text_enc.text_encoder(input_ids=inp.input_ids, attention_mask=inp.attention_mask)
            n_tokens = int(inp.attention_mask[0].sum().item())
            feat = out[0][0, :n_tokens].cpu().float().numpy()  # (n_tokens, 768)
            np.savez(out_path, last_hidden_state=feat)


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--audio_dir', help='directory of <vid>.wav files')
    src.add_argument('--tar_dirs', nargs='+', help='Clotho-Moment WebDataset tar directories')
    parser.add_argument('--jsonl', nargs='+', required=True, help='annotation files whose queries are encoded')
    parser.add_argument('--out_audio', required=True)
    parser.add_argument('--out_text', required=True)
    parser.add_argument('--m2d_dir', default='m2d', help='clone of https://github.com/nttcslab/m2d')
    parser.add_argument('--ckpt', default='m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = get_m2d_model(args.m2d_dir, args.ckpt, device)

    if args.audio_dir:
        extract_audio_dir(args.audio_dir, args.out_audio, model, device, args.batch_size)
    else:
        extract_audio_tars(args.tar_dirs, args.out_audio, model, device, args.batch_size)
    extract_text(args.jsonl, args.out_text, model, device)


if __name__ == '__main__':
    main()
