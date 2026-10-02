"""Load private experiment metadata without scientific-library imports or writes."""
import json
from pathlib import Path

FINAL_INFO_COLUMNS = (
    'pat_id', 'pat', 'model4', 'model15', 'model30', 'model45', 'model60',
    'model90', 'model120', 'model150', 'model180', 'thd45_opt',
)


def load_experiment_settings(filename=None):
    """Return empty collections unless an explicit private JSON file is supplied.

    participants: ordered objects with pat (string) and days (positive number).
    final_info: objects with pat_id (integer), pat (string), and optional saved
        model/threshold fields. Additional legacy column names are preserved.
    pid_xray: integer participant IDs for optional image-based analyses.
    xray_files: participant-ID strings mapped to electrodes/image file paths.
    config: experiment CONFIG overrides, including required preprocessing keys.
    """
    settings = dict(participants=[], final_info=[], pid_xray=[], xray_files={}, config={})
    if filename is None:
        return settings
    path = Path(filename).expanduser()
    with path.open(encoding='utf-8') as handle:
        supplied = json.load(handle)
    if not isinstance(supplied, dict):
        raise ValueError('Experiment settings must be a JSON object')
    unknown = supplied.keys() - settings.keys()
    if unknown:
        raise ValueError('Unknown experiment settings: ' + ', '.join(sorted(unknown)))
    settings.update(supplied)
    for key in ('participants', 'final_info', 'pid_xray'):
        if not isinstance(settings[key], list):
            raise ValueError(f'{key} must be a list')
    for key in ('config', 'xray_files'):
        if not isinstance(settings[key], dict):
            raise ValueError(f'{key} must be an object')
    names = []
    for row in settings['participants']:
        if (not isinstance(row, dict) or not isinstance(row.get('pat'), str)
                or not row['pat'] or not isinstance(row.get('days'), (int, float))
                or isinstance(row['days'], bool) or row['days'] <= 0):
            raise ValueError('Each participants row requires pat and positive days')
        names.append(row['pat'])
    if len(names) != len(set(names)):
        raise ValueError('Participant names must be unique')
    ids, selected_names = [], []
    for row in settings['final_info']:
        if (not isinstance(row, dict) or type(row.get('pat_id')) is not int
                or not isinstance(row.get('pat'), str) or not row['pat']):
            raise ValueError('Each final_info row requires integer pat_id and string pat')
        ids.append(row['pat_id'])
        selected_names.append(row['pat'])
    if len(ids) != len(set(ids)) or len(selected_names) != len(set(selected_names)):
        raise ValueError('final_info participant IDs and names must be unique')
    if names and not set(selected_names).issubset(names):
        raise ValueError('final_info names must appear in participants when supplied')
    if any(type(pid) is not int for pid in settings['pid_xray']):
        raise ValueError('pid_xray must contain integer IDs')
    for files in settings['xray_files'].values():
        if (not isinstance(files, dict) or set(files) != {'electrodes', 'image'}
                or any(not isinstance(p, str) or not p for p in files.values())):
            raise ValueError('Each xray_files entry requires electrodes and image paths')
    return settings
