"""Read-only source preflight. Does not import research modules or load data."""
import argparse
import ast
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from experiment_settings import load_experiment_settings

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {'freq', 'n_chn', 'rec_len_min', 'detrend'}


def config_keys(source):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == 'CONFIG' for t in node.targets
        ):
            return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    return set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--syntax-only', action='store_true', help='Check Python syntax only')
    parser.add_argument('--config', help='Private experiment JSON file to inspect')
    args = parser.parse_args()
    paths = sorted((ROOT / 'code').glob('*.py')) + sorted((ROOT / 'tools').glob('*.py')) + sorted((ROOT / 'tests').glob('*.py'))
    paths += sorted((ROOT / 'paper_reference').rglob('*.py'))
    for path in paths:
        ast.parse(path.read_text(), filename=str(path))
    print(f'Syntax OK: {len(paths)} Python files (no imports or patient data read).')
    if args.syntax_only:
        return 0
    keys = config_keys((ROOT / 'code/config.py').read_text())
    if args.config:
        keys.update(load_experiment_settings(args.config)['config'])
    missing = REQUIRED - keys
    if missing:
        print('BLOCKED: CONFIG is missing preprocessing fields: ' + ', '.join(sorted(missing)))
    print('For dependency and private-input requirements, see docs/setup.md.')
    return 1 if missing else 0


if __name__ == '__main__':
    raise SystemExit(main())

