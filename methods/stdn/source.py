"""STDN executable source closure (root config.py is not the runtime config)."""
from methods.common.learned import verify_source

REQUIRED_FILES = ('README.md', 'train.py', 'test.py', 'model/config.py',
                  'model/model.py', 'model/dataset.py', 'model/warp.py',
                  'model/loss.py', 'model/utils.py')


def validate_source(config, *, source_root=None):
    return verify_source(config, REQUIRED_FILES, source_root=source_root)
