"""Render the five recorded 50 ms gravity-referenced simulations."""
import argparse
import csv
from hashlib import sha256
from pathlib import Path
import shutil

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from ball_study import DEFAULT_OUT, SHAPES, TASKS, trial


def mark_completed_miss(path):
    """Distinguish a missed numerical target after full duration from an early stop."""
    replacement = path.with_suffix('.labeled.mp4')
    with imageio.get_reader(path) as reader, imageio.get_writer(replacement, fps=24, codec='libx264') as writer:
        total = reader.count_frames()
        for i, frame in enumerate(reader):
            if i >= total - 48:
                image = Image.fromarray(frame)
                draw = ImageDraw.Draw(image)
                draw.rectangle((0,46,640,90), fill=(255,244,232))
                draw.text((12,56), 'Completed outside the task criterion', fill=(135,30,22))
                frame = np.asarray(image)
            writer.append_data(frame)
    replacement.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT/'videos')
    parser.add_argument('--publish-dir', type=Path, help='optional second folder for the verified MP4s')
    args = parser.parse_args()
    reference = Path(__file__).resolve().parent/'results'/'gravity.csv'
    with reference.open(newline='') as f:
        rows = {r['task']: r for r in csv.DictReader(f) if r['method']=='mpc' and r['seed']=='401'
                and r['case']=='noisy_50ms' and float(r['sliding_friction'])==0.3
                and float(r['ball_inertia_factor'])==0.4}
    assert set(rows)==set(TASKS)
    decoded = []
    for task in TASKS:
        job = ('mpc', task, 'noisy_50ms', 401, (.018,-.012), .3, .4)
        row = trial(job, suite='gravity', out=args.output_dir, capture_video=True)
        expected = rows[task]
        metric = 'shape_tracking_rms_m' if task in SHAPES else 'final_second_max_center_error_m'
        assert row['performance_pass'] == (expected['performance_pass']=='True')
        value = row[metric]
        if expected[metric]:
            assert value is not None and np.isclose(value, float(expected[metric]), rtol=1.e-6, atol=1.e-8)
        else:
            assert value is None
        assert (row['stop_reason'] or '') == expected['stop_reason']
        paths = list((args.output_dir/'gravity').glob(f'mpc_{task}_noisy_50ms_401_mu0.3_k0.400000/{task}.mp4'))
        path, = paths
        if row['completed'] and not row['performance_pass']:
            mark_completed_miss(path)
        frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
        assert frames > 0 and abs(seconds - frames/24) < .05
        digest = sha256(path.read_bytes()).hexdigest()
        target = (args.publish_dir or args.output_dir)/path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target.read_bytes()).hexdigest()!=digest:
            raise FileExistsError(f'{target} already contains a different video; choose another output folder')
        shutil.copyfile(path,target)
        assert sha256(target.read_bytes()).hexdigest()==digest
        decoded.append(dict(task=task, file=target.name, frames=frames, duration_s=seconds, sha256=digest,
                            performance_pass=row['performance_pass'], metric=metric, error_mm=1000*value if value is not None else None))
        print(task, row['performance_pass'], metric, value, flush=True)
    with (args.output_dir/'video_index.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(decoded[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(decoded)


if __name__=='__main__':
    main()
