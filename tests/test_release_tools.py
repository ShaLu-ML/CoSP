import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'tools'))
from runtime_paths import explicit_paths
from check_readiness import config_keys, REQUIRED


class PathTests(unittest.TestCase):
    def test_no_explicit_settings_returns_none(self):
        self.assertIsNone(explicit_paths({}))

    def test_partial_settings_fail(self):
        with self.assertRaisesRegex(ValueError, 'COSP_WORK_PATH'):
            explicit_paths({'COSP_DATA_PATH': '/tmp'})

    def test_explicit_paths_do_not_create_work_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / 'new-work'
            env = dict(COSP_DATA_PATH=tmp, COSP_WORK_PATH=str(work),
                       COSP_FILE_LEN_MIN='10', COSP_REC_OFFSET='500')
            self.assertEqual(explicit_paths(env), (Path(tmp), work, 10, 500))
            self.assertFalse(work.exists())
            for key, value in [('COSP_FILE_LEN_MIN', '0'), ('COSP_REC_OFFSET', '-1'),
                               ('COSP_DATA_PATH', 'relative'), ('COSP_DATA_PATH', str(work))]:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    explicit_paths({**env, key: value})

    def test_config_inspection_never_executes_source(self):
        self.assertEqual(config_keys("raise RuntimeError()\nCONFIG = {'freq': 400}"), {'freq'})
        self.assertEqual(REQUIRED - config_keys("CONFIG = {'freq': 400}"),
                         {'n_chn', 'rec_len_min', 'detrend'})


if __name__ == '__main__':
    unittest.main()

