import argparse
import csv
from datetime import datetime
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]


def jobs(spec):
    grid = spec.get('grid', {})
    grid_keys = list(grid.keys())
    grid_values = list(itertools.product(*(grid[key] for key in grid_keys)))
    cases = spec.get('cases', [{
        'name': '', 'overrides': [], 'train_fractions': spec.get('train_fractions', []),
    }])
    for variant, task, seed, case in itertools.product(spec.variants, spec.tasks, spec.seeds, cases):
        if task not in variant.get('tasks', spec.tasks):
            continue
        for fraction, values in itertools.product(case['train_fractions'], grid_values):
            name = f"{task}_{variant.name}_seed{seed}_fraction{fraction}"
            if case['name']:
                name += '_' + case['name']
            for key, value in zip(grid_keys, values):
                name += f"_{key}_{value}"
            if not re.fullmatch(r"[a-zA-Z0-9_.-]+", name):
                raise ValueError(f"Invalid run name: {name}")
            overrides = [f"task={task}", *spec.common, *variant.overrides, *case['overrides'],
                         f"seed={seed}", f"data.train_fraction={fraction}",
                         *[f"{key}={value}" for key, value in zip(grid_keys, values)]]
            yield name, str(variant.name), overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'sweeps/data_fraction_modulus.yaml')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    spec = OmegaConf.load(args.config)
    planned = list(jobs(spec))
    if not planned or len({name for name, _, _ in planned}) != len(planned):
        raise ValueError('Sweep must have nonempty, uniquely named jobs')
    output_root = (args.output_dir or ROOT / 'runs' / ('sweep_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if args.dry_run:
        for name, _, overrides in planned:
            print(name + ': python train.py ' + ' '.join(overrides))
        print(f'{len(planned)} runs')
        return
    output_root.mkdir(parents=True, exist_ok=False)
    results_path = output_root / 'results.csv'
    columns = ['run', 'epoch', 'test_exact_match', 'test_digit_accuracy',
               'test_loss', 'test_reward', 'training_seconds']
    with results_path.open('w', newline='', encoding='utf-8') as file:
        csv.DictWriter(file, fieldnames=columns).writeheader()
    print(f'Results: {results_path}', flush=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for i, (name, _, overrides) in enumerate(planned, 1):
        directory = output_root / name
        directory.mkdir()
        print(f'[{i}/{len(planned)}] {name}', flush=True)
        command = [sys.executable, str(ROOT / 'train.py'), *overrides,
                   f'hydra.run.dir={directory.as_posix()}']
        with (directory / 'console.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Run failed ({result.returncode}); see {directory / "console.log"}')
        metrics = [json.loads(line) for line in
                   (directory / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        if not (directory / 'model.pt').exists():
            raise RuntimeError(f'Run has no final checkpoint: {directory}')
        final = metrics[-1]
        row = {'run': name, **{key: final.get(key, '') for key in columns[1:]}}
        # Save each completed run immediately, including when a later run fails.
        with results_path.open('a', newline='', encoding='utf-8') as file:
            csv.DictWriter(file, fieldnames=columns).writerow(row)
        print(f"  exact={final['test_exact_match']:.1%}, digits={final['test_digit_accuracy']:.1%}", flush=True)


if __name__ == '__main__':
    main()
