"""DSDG-BIN-IDFREE: pinned models/losses, frozen A1 pair/label adaptation.

No image reader or training launch is invoked by this module's preparation API.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import re

import numpy as np
import yaml

from methods.common.config import ROOT, load_method_config, sha256_file
from methods.common.learned import (authoritative, checkpoint_plan, effective_batch,
                                    environment_report, mapping, PreparationError,
                                    seed_plan, verify_asset)
from methods.common.upstream import upstream_modules
from .source import validate_source

ARG_FIELDS = {
    '--workers': 'training.workers', '--all_epochs': 'training.all_epochs',
    '--pre_epoch': 'training.pre_epoch', '--hdim': 'training.hdim',
    '--attack_type': 'training.attack_type', '--test_epoch': 'training.test_epoch',
    '--lr': 'optimizer.learning_rate', '--lambda_mmd': 'losses.lambda_mmd',
    '--lambda_ip': 'losses.lambda_ip', '--lambda_pair': 'losses.lambda_pair',
    '--lambda_type': 'losses.lambda_type', '--lambda_ort': 'losses.lambda_ort',
}


class DSDGAdapter:
    method_id = 'E06c'

    def __init__(self, config=None, *, source_root=None):
        self.config = authoritative(load_method_config(self.method_id) if config is None else config)
        if self.config['method_id'] != self.method_id:
            raise PreparationError('DSDG ID-free requires E06c config')
        self.source_root = source_root
        self.semantics = self._semantics()

    def _semantics(self):
        cfg = authoritative(self.config)
        ref = cfg['adaptation_semantics']
        path = ROOT / ref['frozen_in']
        if sha256_file(path) != ref['frozen_config_sha256']:
            raise PreparationError('A1 adaptation bytes differ from frozen hash')
        sem = yaml.safe_load(path.read_text())
        if cfg['losses']['lambda_pair'] != sem['losses']['lambda_pair'] or cfg['losses']['lambda_pair'] != 0:
            raise PreparationError('A1 prohibits pair-identity loss')
        if cfg['training']['attack_type'] != sem['binary_supervision']['attack_type_arg']:
            raise PreparationError('binary spoof classification must have one category')
        if cfg['subject_id_global_consumed'] or any(
                v is not False for k, v in sem['identity_firewall'].items()
                if k.startswith('consumed_for_') or k == 'subject_id_global_consumed'):
            raise PreparationError('identity supervision is forbidden')
        if sem['training_relation']['authoritative_manifest_sha256'] != cfg['data']['training_relation_sha256']:
            raise PreparationError('A1/M6B training relation disagreement')
        return sem

    def validate_source(self):
        return validate_source(self.config, source_root=self.source_root)

    def validate_environment(self):
        return environment_report('torch', ('torch', 'torchvision', 'numpy', 'PIL', 'cv2'))

    def verify_lightcnn(self):
        cfg = authoritative(self.config)
        asset, = cfg['external_assets']
        result = verify_asset(asset)
        provenance = json.loads((ROOT / asset['provenance']).read_text())
        prior = provenance['local_file']
        if (prior['sha256'] != result['sha256'] or prior['byte_size'] != result['size_bytes'] or
                prior['local_external_runtime_path'] != result['path']):
            raise PreparationError('LightCNN provenance/config mismatch')
        compatibility = provenance['compatibility_verification']
        if not compatibility['compatible'] or not compatibility['matched_subset_is_complete']:
            raise PreparationError('LightCNN historical compatibility not established')
        result['compatibility'] = compatibility
        result['compatibility_basis'] = 'Same bytes as M6A4 restricted inspection; not re-deserialized'
        return result

    def prepare(self):
        self.semantics = self._semantics()
        self.source = self.validate_source()
        self.lightcnn = self.verify_lightcnn()
        self.settings = mapping(self.config, ARG_FIELDS, 'addition_module/DSDG/train_generator.py:parser')
        cadence = re.fullmatch(r'save_epoch_(\d+)', self.config['checkpoint']['cadence'])
        if cadence is None:
            raise PreparationError('unrecognized frozen official save cadence')
        self.settings.append({'target': '--save_epoch', 'value': int(cadence[1]),
                              'config_field': 'checkpoint.cadence',
                              'source_file': 'addition_module/DSDG/train_generator.sh'})
        self.settings.append({'target': '--ip_model', 'value': self.lightcnn['path'],
                              'config_field': 'external_assets[0].external_runtime_path',
                              'source_file': 'addition_module/DSDG/train_generator.py:define_IP loading'})
        return self

    def binary_label(self, label):
        """Benchmark live/spoof labels, separate from the single-spoof CE index."""
        if label not in ('live', 'spoof'):
            raise PreparationError('only binary live/spoof labels are accepted')
        return ('live', 'spoof').index(label)

    def project_pair(self, record):
        """Allowlist projection: subject/attack columns are never inspected or forwarded."""
        sem = self.semantics
        fields = sem['training_relation']['columns_consumed']
        required = sem['identity_firewall']['adapter_interface_fields']
        if any(k not in record for k in required):
            raise PreparationError('required frozen pair fields missing')
        result = {k: record[k] for k in fields if k in record}
        if result['split'] != sem['training_relation']['split']:
            raise PreparationError('DSDG training pairs must be TRAIN only')
        if result['dataset'] not in sem['datasets']:
            raise PreparationError('pair dataset is outside A1')
        if not all(isinstance(result[k], str) and result[k] for k in
                   ('pair_id', 'source_spoof_id', 'target_live_id')):
            raise PreparationError('pair/sample IDs must be nonempty strings')
        return result

    def adapt_pair_arrays(self, record, *, source_rgb, target_rgb):
        """In-memory adapter seam for already canonical RGB arrays; does no I/O.

        Official GenDataset_s yields keys 0=spoof, 1=live, type=CE index.
        The fixed pair replaces ONLY the identity-based sampler/crop interface.
        """
        self.project_pair(record)
        size = self.config['training']['input_resolution']
        def convert(array):
            array = np.asarray(array)
            if array.shape != (size, size, 3) or array.dtype != np.uint8:
                raise PreparationError('canonical uint8 RGB256 required; no implicit recrop/rescale')
            return np.ascontiguousarray(array.transpose(2, 0, 1), dtype=np.float32) / 255
        # A single spoof category has zero-based class index 0, NOT binary spoof label 1.
        return {'0': convert(source_rgb), '1': convert(target_rgb),
                'type': np.int64(self.config['training']['attack_type'] - 1)}

    def validate_batch(self, *, physical_batch_size=None, gradient_accumulation_steps=1,
                       replica_factor=1, oom_reason=None):
        """Arithmetic equality alone cannot preserve the pinned batch-dependent losses.

        train_generator.py::main, lines 141 and 147, apply abs AFTER batch means
        for MMD and angular orthogonality. No equivalent microbatch loss/backward
        implementation is demonstrated here. An OOM reason is not an override.
        """
        cfg = authoritative(self.config)
        required = cfg['training']['effective_batch_size']
        physical = required if physical_batch_size is None else physical_batch_size
        policy = 'E06C_REQUIRES_PHYSICAL_BATCH_240_FOR_OBJECTIVE_EQUIVALENCE'
        if (physical, gradient_accumulation_steps, replica_factor) != (required, 1, 1):
            raise PreparationError(
                f'{policy}: gradient accumulation does not preserve the frozen '
                'batch-dependent objective automatically (MMD and angular orthogonality). '
                f'E06c requires physical_batch_size={required}, '
                'gradient_accumulation_steps=1, replica_factor=1. '
                'OOM behavior: STOP_AND_RESOLVE; an OOM reason alone cannot bypass this invariant.')
        batch = effective_batch(cfg, physical_batch_size=physical,
                                gradient_accumulation_steps=gradient_accumulation_steps,
                                replica_factor=replica_factor, oom_reason=oom_reason)
        batch.update(
            execution_policy=policy, oom_behavior='STOP_AND_RESOLVE',
            naive_accumulation_objective_equivalent=False,
            semantics_preserving_microbatch_implementation_demonstrated=False,
            batch_dependent_losses={
                'loss_mmd': 'addition_module/DSDG/train_generator.py::main:141',
                'loss_ort': 'addition_module/DSDG/train_generator.py::main:147'},
            loss_accounting='Use the official full physical forward batch. No microbatch '
                            'accumulation fallback; OOM requires an execution-resolution decision.')
        return batch

    def build_training_plan(self, seed, *, selection_split=None, **batch_options):
        batch = self.validate_batch(**batch_options)
        self.prepare()
        cfg = self.config
        settings = self.settings + [{'target': '--batch_size', 'value': batch['physical_batch_size'],
                                    'config_field': 'training.effective_batch_size; E06c requires the full physical batch',
                                    'source_file': 'addition_module/DSDG/train_generator.py:DataLoader'}]
        return {'method_id': self.method_id, 'status': 'STATIC_PREPARED_NOT_EXECUTED',
                'experiment_seed': seed, 'config_sha256': cfg['_runtime']['config_sha256'],
                'snapshot_verified': True, 'fidelity_class': cfg['fidelity_class'],
                'fidelity_provenance': cfg['fidelity_provenance'], 'source': self.source,
                'lightcnn': self.lightcnn, 'settings_mapping': settings,
                'official_argument_vector': [s for item in settings for s in (item['target'], str(item['value']))],
                'argument_vector_is_launch_command': False,
                'optimizer_contract': cfg['optimizer'], 'loss_contract': cfg['losses'],
                'seed_plan': seed_plan(cfg, seed, 'torch'), 'effective_batch': batch,
                'checkpoint': checkpoint_plan(cfg, seed, selection_split=selection_split),
                'environment': self.validate_environment(),
                'upstream_bindings': {'models': 'networks.define_G', 'identity_net': 'networks.define_IP',
                                     'loss_loop': 'train_generator.main (pinned body)',
                                     'optimizer': 'torch.optim.Adam(netE_nir+netE_vis+netG)',
                                     'checkpoint_writer': 'misc.util.save_checkpoint'},
                'adaptation': {'contract': cfg['adaptation_semantics'],
                               'training_relation': cfg['data']['training_relation'],
                               'consumed_columns': self.semantics['training_relation']['columns_consumed'],
                               'identity_supervision': False, 'lambda_pair': cfg['losses']['lambda_pair'],
                               'binary_labels': {'live': self.binary_label('live'), 'spoof': self.binary_label('spoof')},
                               'spoof_cross_entropy_index': cfg['training']['attack_type'] - 1,
                               'loss_cls': self.semantics['binary_supervision']['degeneracy_disclosure'],
                               'sampler': 'methods.dsdg.adapter.IDFreePairDataset replaces GenDataset_s same-subject sampling',
                               'loader_rng': 'methods.common.learned.torch_loader_options',
                               'test_code_path_present': False},
                'execution_notes': ['Do not invoke unadapted upstream main: it uses identity pairs and unseeded workers.',
                                    'Future runner binds projected pair records, seeded DataLoader generator/worker hook and learned logs.',
                                    'GPU IDs/output directories are runtime resources, not scientific defaults.',
                                    'Official periodic test block is training-sample visualization, not validation.',
                                    'Upstream CUDA-only reparameterize/define_G requires a compatible GPU environment.',
                                    'No standalone training launcher or OOM workaround is executed in M6C2a.'],
                'training_launched': False, 'checkpoint_created': False}

    @contextmanager
    def official_components(self):
        self.prepare()
        env = self.validate_environment()
        if env['missing_modules']:
            raise PreparationError(env['reason'])
        root = Path(self.source['root']) / self.config['source']['relevant_path']
        with upstream_modules(root, ('train_generator', 'networks', 'misc'),
                              ('train_generator', 'networks', 'misc', 'data')) as modules:
            yield modules


class IDFreePairDataset:
    """Future DataLoader-compatible relation adapter, with injected canonical reader.

    Takes already verified manifest records; does not read/manufacture a manifest.
    The reader receives sample IDs only, never identity/attack metadata. During
    M6C2a tests it is an in-memory dictionary lookup on synthetic IDs.
    """
    def __init__(self, adapter, records, canonical_rgb_reader):
        self.adapter = adapter
        self.records = tuple(adapter.project_pair(row) for row in records)
        self.reader = canonical_rgb_reader

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return self.adapter.adapt_pair_arrays(
            row, source_rgb=self.reader(row['source_spoof_id']),
            target_rgb=self.reader(row['target_live_id']))
