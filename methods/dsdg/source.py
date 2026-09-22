"""The generator training import closure of pinned FaceX-Zoo DSDG."""
from methods.common.learned import verify_source

PREFIX = 'addition_module/DSDG/'
REQUIRED_FILES = tuple(PREFIX + f for f in (
    'README.md', 'train_generator.sh', 'train_generator.py', 'generated.py',
    'data/__init__.py', 'data/generation_dataset.py', 'data/recognition_dataset.py',
    'networks/__init__.py', 'networks/generator.py', 'networks/light_cnn.py',
    'misc/__init__.py', 'misc/util.py'))


def validate_source(config, *, source_root=None):
    return verify_source(config, REQUIRED_FILES, source_root=source_root)
