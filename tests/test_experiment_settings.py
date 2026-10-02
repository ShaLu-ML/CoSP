"""Synthetic settings and data-free export checks."""
import ast
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from experiment_settings import load_experiment_settings, FINAL_INFO_COLUMNS


class ExperimentSettingsTests(unittest.TestCase):
    def load(self, data):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'settings.json'
            path.write_text(json.dumps(data))
            return load_experiment_settings(path)

    def test_empty_metadata(self):
        self.assertEqual(load_experiment_settings()['final_info'], [])
        self.assertIn('model45', FINAL_INFO_COLUMNS)
        self.assertEqual(self.load({})['participants'], [])

    def test_synthetic_metadata_and_extra_columns(self):
        data = dict(participants=[dict(pat='synthetic_subject', days=1)],
                    final_info=[dict(pat_id=9001, pat='synthetic_subject', custom_model=9002)],
                    pid_xray=[9001], config={'pat_id': 9001},
                    xray_files={'9001': {'electrodes': 'synthetic.mat', 'image': 'synthetic.tif'}})
        self.assertEqual(self.load(data), data)

    def test_invalid_schema(self):
        bad = [[], {'unknown': []}, {'participants': {}}, {'config': []},
               {'participants': [{'pat': 'synthetic_subject', 'days': 0}]},
               {'final_info': [{'pat_id': '9001', 'pat': 'synthetic_subject'}]},
               {'final_info': [{'pat_id': 9001, 'pat': 'a'}, {'pat_id': 9001, 'pat': 'b'}]},
               {'participants': [{'pat': 'a', 'days': 1}], 'final_info': [{'pat_id': 9001, 'pat': 'b'}]},
               {'pid_xray': ['9001']}, {'xray_files': {'9001': {'image': 'synthetic.tif'}}}]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load(value)

    def test_missing_private_file_is_explicit(self):
        with self.assertRaises(FileNotFoundError):
            load_experiment_settings('/this/private/config/does/not/exist.json')

    def test_template_has_no_experiment_rows(self):
        template = load_experiment_settings(ROOT / 'examples/experiment.example.json')
        self.assertEqual(template, load_experiment_settings())


class SourceChecks(unittest.TestCase):
    def test_no_bundled_data_or_git_history(self):
        forbidden = {'.csv', '.tsv', '.mat', '.pkl', '.pickle', '.npy', '.npz', '.pth', '.pt', '.edf'}
        for path in ROOT.rglob('*'):
            self.assertNotIn(path.suffix.lower(), forbidden)
        # A fresh clone can have its own Git history; no old object store is bundled.
        self.assertFalse((ROOT / 'result').exists())

    def test_config_uses_private_metadata(self):
        tree = ast.parse((ROOT / 'code/config.py').read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {'PAT_LIST_ALL', 'PAT_ALL_DAYS', 'PID_XRAY', 'FINAL_INFO'}:
                        self.assertNotIsInstance(node.value, (ast.List, ast.Dict))

    def test_analysis_inputs_are_explicit(self):
        tree = ast.parse((ROOT / 'code/analysis.py').read_text())
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertEqual([a.arg for a in functions['wilcox_test'].args.args], ['cosp_scores', 'compare_scores'])
        self.assertEqual(functions['plot_results_histogram'].args.args[0].arg, 'data')
        for name in ['wilcox_test', 'plot_results_histogram']:
            for node in ast.walk(functions[name]):
                if isinstance(node, ast.List) and len(node.elts) >= 8:
                    self.assertFalse(all(isinstance(x, ast.Constant) and isinstance(x.value, (int, float)) for x in node.elts))

    def test_no_private_launch_blocks(self):
        for path in (ROOT / 'code').glob('*.py'):
            source = path.read_text()
            self.assertNotIn("if __name__ == '__main__'", source)
            self.assertNotRegex(source, r'/(?:home|ssd2|data/projects|data/gpfs)/')


if __name__ == '__main__':
    unittest.main()
