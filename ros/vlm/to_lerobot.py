#!/usr/bin/env python3
"""Wandelt aufgezeichnete Episoden (episode_recorder.py) in ein LeRobot-Dataset um.

  pixi run -e lerobot to-lerobot                    # nur erfolgreiche Episoden
  pixi run -e lerobot to-lerobot --all --depth      # alle, mit Tiefenbild

Läuft in der eigenen pixi-Umgebung „lerobot“ (torch, datasets …), nicht in der
ROS-Umgebung. Zuordnung pro Schritt (10 Hz):

  observation.images.front   rgb/NNNNNN.jpg            640×480, Video (AV1)
  observation.images.depth   depth/NNNNNN.png          320×240, uint16 mm,
                             (nur mit --depth)         Tiefenvideo (HEVC 12 bit)
  observation.state          [x, y, yaw, v, ω, steering]  state + velocity + steering
  action                     [v, ω]                    befohlenes /cmd_vel
  task                       meta.json → instruction

Fehlende Werte (NaN, z. B. bevor AMCL eine Pose liefert) werden pro Spalte mit
dem nächsten gültigen Wert aufgefüllt. Woher jede LeRobot-Episode stammt
(Episoden-ID, Aufgabe, Erfolg …), steht in meta/rover_episodes.jsonl.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STATE_NAMES = ['x', 'y', 'yaw', 'v', 'w', 'steering']
ACTION_NAMES = ['v', 'w']
RGB_SHAPE = (480, 640, 3)
DEPTH_SHAPE = (240, 320, 1)
DEPTH_MAX_M = 12.0  # <far> der Tiefenkamera in models/rover/model.sdf
# torchcodec findet unter macOS ohne systemweites FFmpeg seine Bibliotheken nicht
VIDEO_BACKEND = 'pyav'


def succeeded(meta):
    """Erfolg laut Ground Truth, sonst laut Agent (None = unbekannt)."""
    gt = meta.get('gt_success')
    return gt if gt is not None else meta.get('success')


def fill_nan(a):
    """NaN pro Spalte mit dem letzten (am Anfang: dem ersten) gültigen Wert füllen."""
    a = a.copy()
    for col in a.T:
        bad = ~np.isfinite(col)
        if not bad.any():
            continue
        if bad.all():
            col[:] = 0.0
            continue
        idx = np.where(~bad, np.arange(len(col)), 0)
        np.maximum.accumulate(idx, out=idx)
        first = np.argmax(~bad)
        idx[:first] = first
        col[:] = col[idx]
    return a


def load_episodes(src, include_failed):
    episodes, skipped = [], []
    for path in sorted(p for p in src.iterdir() if p.is_dir()):
        if not (path / 'meta.json').exists() or not (path / 'steps.npz').exists():
            skipped.append((path.name, 'unvollständig (meta.json/steps.npz fehlt)'))
            continue
        meta = json.loads((path / 'meta.json').read_text())
        ok = succeeded(meta)
        if not include_failed and ok is not True:
            skipped.append((path.name, f'Erfolg={ok}'))
            continue
        if not meta.get('num_steps'):
            skipped.append((path.name, 'leer'))
            continue
        episodes.append((path, meta))
    return episodes, skipped


def features(depth):
    feats = {
        'observation.images.front': {
            'dtype': 'video', 'shape': RGB_SHAPE, 'names': ['height', 'width', 'channels']},
        'observation.state': {
            'dtype': 'float32', 'shape': (len(STATE_NAMES),), 'names': STATE_NAMES},
        'action': {
            'dtype': 'float32', 'shape': (len(ACTION_NAMES),), 'names': ACTION_NAMES},
    }
    if depth:
        feats['observation.images.depth'] = {
            'dtype': 'video', 'shape': DEPTH_SHAPE, 'names': ['height', 'width', 'channels'],
            'info': {'is_depth_map': True}}
    return feats


def convert_episode(ds, path, meta, depth):
    steps = np.load(path / 'steps.npz')
    n = len(steps['t'])
    n_rgb = len(list((path / 'rgb').glob('*.jpg')))
    if n_rgb != n:
        print(f'  {path.name}: {n} Schritte, aber {n_rgb} Bilder – nehme {min(n, n_rgb)}')
        n = min(n, n_rgb)
    state = np.concatenate([steps['state'][:n], steps['velocity'][:n],
                            steps['steering'][:n, None]], axis=1)
    nan_count = int((~np.isfinite(state)).sum())
    state = fill_nan(state).astype(np.float32)
    action = np.nan_to_num(steps['action'][:n]).astype(np.float32)
    task = meta.get('instruction') or meta.get('task') or ''
    missing_depth = 0
    for i in range(n):
        bgr = cv2.imread(str(path / 'rgb' / f'{i:06d}.jpg'), cv2.IMREAD_COLOR)
        frame = {
            'observation.images.front': cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            'observation.state': state[i],
            'action': action[i],
            'task': task,
        }
        if depth:
            d = cv2.imread(str(path / 'depth' / f'{i:06d}.png'), cv2.IMREAD_UNCHANGED)
            if d is None:  # Recorder schreibt Tiefe nur, wenn schon ein Bild da war
                d = np.zeros(DEPTH_SHAPE[:2], np.uint16)
                missing_depth += 1
            frame['observation.images.depth'] = d.astype(np.uint16)[..., None]
        ds.add_frame(frame)
    ds.save_episode()
    notes = []
    if nan_count:
        notes.append(f'{nan_count} NaN im Zustand aufgefüllt')
    if missing_depth:
        notes.append(f'{missing_depth} Tiefenbilder fehlten (0 = ungültig)')
    return n, notes


def verify(repo_id, out, expected_frames, fps):
    """Dataset wie ein Trainingsskript laden und Bild/Zustand stichprobenartig lesen."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=out, video_backend=VIDEO_BACKEND)
    assert ds.fps == fps, f'fps {ds.fps} != {fps}'
    assert ds.num_episodes == len(expected_frames), f'{ds.num_episodes} Episoden != {len(expected_frames)}'
    assert ds.num_frames == sum(expected_frames), f'{ds.num_frames} Frames != {sum(expected_frames)}'
    for ep, n in enumerate(expected_frames):
        ep_meta = ds.meta.episodes[ep]
        got = ep_meta['dataset_to_index'] - ep_meta['dataset_from_index']
        assert got == n, f'Episode {ep}: {got} Frames != {n}'
    for idx in (0, ds.num_frames - 1):
        item = ds[idx]
        for key in ds.meta.camera_keys:
            assert item[key].ndim == 3, f'{key}: Form {tuple(item[key].shape)}'
        assert item['observation.state'].shape == (len(STATE_NAMES),)
        assert item['action'].shape == (len(ACTION_NAMES),)
    item = ds[0]
    print(f'geladen: {ds.num_episodes} Episoden, {ds.num_frames} Frames, {ds.fps} fps; '
          f'Aufgabe 0: „{item["task"]}“; '
          + ', '.join(f'{k} {tuple(item[k].shape)}' for k in ds.meta.camera_keys))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--src', type=Path, default=ROOT / 'data' / 'episodes',
                    help='Ordner mit den Episoden (Standard: data/episodes)')
    ap.add_argument('--out', type=Path, default=ROOT / 'data' / 'lerobot' / 'rover_office',
                    help='Zielordner des Datasets (Standard: data/lerobot/rover_office)')
    ap.add_argument('--repo-id', default='local/rover_office',
                    help='Name für LeRobot/Hugging Face Hub (Standard: local/rover_office)')
    ap.add_argument('--all', action='store_true',
                    help='auch fehlgeschlagene Episoden übernehmen (Standard: nur gt_success/success)')
    ap.add_argument('--depth', action='store_true', help='Tiefenbild als observation.images.depth')
    ap.add_argument('--fps', type=int, default=10, help='Aufnahmerate (Standard: 10)')
    ap.add_argument('--vcodec', default='libsvtav1',
                    help='Codec für die Farbkamera (z. B. libsvtav1, h264, h264_videotoolbox)')
    ap.add_argument('--overwrite', action='store_true', help='vorhandenes Dataset ersetzen')
    args = ap.parse_args()

    episodes, skipped = load_episodes(args.src, args.all)
    for name, why in skipped:
        print(f'übersprungen {name}: {why}')
    wrong_rate = [p.name for p, m in episodes if m.get('rate_hz', args.fps) != args.fps]
    if wrong_rate:
        sys.exit(f'Episoden mit anderer Rate als {args.fps} Hz: {wrong_rate}')
    if not episodes:
        sys.exit(f'keine passenden Episoden in {args.src}')
    if args.out.exists():
        if not args.overwrite:
            sys.exit(f'{args.out} existiert schon (--overwrite zum Ersetzen)')
        shutil.rmtree(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    from lerobot.configs.video import DepthEncoderConfig, RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps, root=args.out, robot_type='ackermann_rover',
        features=features(args.depth), video_backend=VIDEO_BACKEND,
        rgb_encoder=RGBEncoderConfig(vcodec=args.vcodec),
        depth_encoder=DepthEncoderConfig(depth_max=DEPTH_MAX_M),
        # Bilder pro Episode zwischenspeichern, dann kodieren; streaming_encoding
        # verwirft Frames, sobald der Encoder nicht mitkommt
        image_writer_threads=4)

    frames, sources = [], []
    for ep_index, (path, meta) in enumerate(episodes):
        n, notes = convert_episode(ds, path, meta, args.depth)
        frames.append(n)
        sources.append({'episode_index': ep_index, 'source_id': path.name, 'frames': n,
                        **{k: meta.get(k) for k in ('instruction', 'task', 'perception', 'model',
                                                    'success', 'gt_success')}})
        print(f'[{ep_index + 1}/{len(episodes)}] {path.name}: {n} Frames, '
              f'„{meta.get("instruction")}“' + (f' ({"; ".join(notes)})' if notes else ''))
    ds.finalize()
    with open(args.out / 'meta' / 'rover_episodes.jsonl', 'w') as f:
        for s in sources:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    print(f'gespeichert: {args.out}')
    verify(args.repo_id, args.out, frames, args.fps)


if __name__ == '__main__':
    main()
