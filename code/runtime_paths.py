"""Optional explicit paths; no scientific defaults or filesystem writes."""
from pathlib import Path


def explicit_paths(environ):
    names = ('COSP_DATA_PATH', 'COSP_WORK_PATH', 'COSP_FILE_LEN_MIN', 'COSP_REC_OFFSET')
    if not any(name in environ for name in names):
        return None
    missing = [name for name in names if not environ.get(name)]
    if missing:
        raise ValueError('Explicit path configuration requires: ' + ', '.join(missing))
    data, work = (Path(environ[name]).expanduser() for name in names[:2])
    if not data.is_absolute() or not work.is_absolute():
        raise ValueError('COSP_DATA_PATH and COSP_WORK_PATH must be absolute')
    file_len, offset = (int(environ[name]) for name in names[2:])
    if file_len < 1 or offset < 0:
        raise ValueError('File length must be positive and record offset nonnegative')
    if not data.is_dir():
        raise ValueError('COSP_DATA_PATH must be an existing directory')
    return data, work, file_len, offset
