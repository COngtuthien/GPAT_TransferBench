"""Isolated Python 3.8+ synthetic E03 worker; no benchmark runner or API shim.

Load this file directly, avoiding the modern controller package's __init__.
All scientific functions are imported from the hash-verified, unchanged source.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

PIN = 'c79f1f8c615d2b8471b3df29da881bb18dd54c90'
PRECISION = {
    'NVIDIA_TF32_OVERRIDE': '0', 'TF_ENABLE_AUTO_MIXED_PRECISION': '0',
    'TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE': '0',
    'TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32': '0',
    'TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32': '0',
    'TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32': '0',
    'TF_XLA_FLAGS': '--tf_xla_auto_jit=0', 'TF_DETERMINISTIC_OPS': '1',
    'TF_CUDNN_DETERMINISTIC': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
    'TF_ENABLE_ONEDNN_OPTS': '0', 'PYTHONDONTWRITEBYTECODE': '1',
    'PYTHONNOUSERSITE': '1', 'CUDA_VISIBLE_DEVICES': '0',
}
REQUIRED_APIS = ('Session', 'ConfigProto', 'GPUOptions', 'py_func',
                 'image.resize_images', 'train.AdamOptimizer', 'ensure_shape',
                 'random.uniform', 'set_random_seed', 'contrib.framework.arg_scope',
                 'contrib.layers.conv2d', 'contrib.layers.conv2d_transpose',
                 'contrib.layers.batch_norm', 'contrib.layers.dropout',
                 'contrib.layers.fully_connected')


def digest(b):
    return hashlib.sha256(b).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()


def under(path, root):
    """Python 3.8-compatible containment; no monkey-patch of pathlib."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


class Firewall:
    def __init__(self, repo, source):
        self.repo, self.source = Path(repo).resolve(), Path(source).resolve()
        self.denied = []

    def __call__(self, event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        p = Path(os.fsdecode(args[0])).absolute()
        flags = args[2] if isinstance(args[2], int) else 0
        write = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        forbidden = (under(p, self.repo / 'manifests') or under(p, self.repo / 'data')
                     or under(p, self.repo / 'cache') or under(p, self.source / 'data')
                     or 'GPAT_TransferBench_runtime' in p.parts
                     or p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.npy', '.npz', '.parquet', '.ckpt', '.h5', '.pth', '.pt'})
        if forbidden or (write and not under(p, '/tmp') and str(p) != '/dev/null'):
            self.denied.append(str(p))
            raise RuntimeError('E03 synthetic firewall refuses ' + str(p))


def expected_variables():
    """Independent architecture oracle, derived from pinned layer definitions."""
    result = {}
    def conv(scope, cin, cout, plain=False, transpose=False):
        result[scope + '/weights:0'] = [3, 3, cout, cin] if transpose else [3, 3, cin, cout]
        result[scope + '/biases:0'] = [cout]
        if not plain:
            for n in ['beta', 'gamma']:
                result[scope + '/BN/' + n + ':0'] = [cout]
            result[scope + '/PRelu/alpha:0'] = [cout]
    for n, cin, cout in [('conv0',6,64),('conv1',64,64),('conv2',64,96),('conv3',96,64),
                         ('conv4',64,64),('conv5',64,96),('conv6',96,64),
                         ('conv7',64,64),('conv8',64,96),('conv9',96,64)]:
        conv('STDN/' + n, cin, cout)
    for n, cin in [('up1',64),('up2',128),('up3',128)]:
        conv('STDN/' + n, cin, 64, transpose=True)
    for n, cout in [('1',6),('2',3),('3',3)]:
        conv('STDN/n' + n, 64, 16)
        conv('STDN/nn' + n, 16, cout, plain=True)
    conv('STDN/conv10',192,64); conv('STDN/conv11',64,64); conv('STDN/conv12',64,1,plain=True)
    for d in ['d1','d2','d3']:
        for n,cin,cout in [('conv2',6,32),('conv4',32,64),('conv6',64,96),('conv7',96,96),('conv8',96,1),('conv9',96,1)]:
            conv('Disc/' + d + '/' + n,cin,cout,plain=n in ['conv8','conv9'])
    return result


def verify_contract(repo, source):
    import yaml
    repo, source = Path(repo), Path(source)
    cfg_bytes = (repo / 'configs/methods/e03_stdn.yaml').read_bytes()
    assert cfg_bytes == (repo / 'frozen_config_snapshot/configs/methods/e03_stdn.yaml').read_bytes()
    cfg = yaml.safe_load(cfg_bytes)
    assert cfg['source']['pinned_commit'] == PIN and cfg['seeds']['experiment_seeds'] == [42,1337,2026]
    assert cfg['data']['splits']['TEST']['allowed'] is False
    assert cfg['checkpoint']['rule'] == 'OFFICIAL_LATEST_FINAL_CKPT_50'
    expected = {'input_resolution':256,'map_size':32,'batch_size':2,'g_d_ratio':2,'max_epoch':50,'steps_per_epoch':2000,'val_steps':500}
    assert all(cfg['training'][k] == v for k,v in expected.items())
    assert cfg['optimizer']['learning_rate'] == 6e-5
    assert [cfg['optimizer'][k] for k in ['beta1','beta2','epsilon','weight_ema_decay','loss_ema_decay']] == [0.9,0.999,1e-8,0.9999,0.9]
    pins = json.loads((repo/'third_party/source_pins.json').read_text())['sources']['stdn']
    prior = json.loads((repo/'outputs/audit/M6D2A_E03_ENVIRONMENT_RESOLUTION.json').read_text())['laptop_source']
    hashes = {}
    for rel,row in prior['files'].items():
        if rel.endswith('.py') or rel == 'README.md':
            b=(source/rel).read_bytes(); assert digest(b)==row['sha256']; hashes[rel]=digest(b)
    for row in pins['binary_weights_removed_after_checkout']:
        assert not (source/row['path']).exists()
    return cfg, hashes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True);p.add_argument('--source',required=True)
    p.add_argument('--stage',choices=['A','B','C','D','E'],required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--optimizer-path',choices=['G','D'],default='G')
    args=p.parse_args()
    assert args.seed in [42,1337,2026]
    for k,v in PRECISION.items(): assert os.environ.get(k)==v,(k,os.environ.get(k))
    assert os.environ.get('PYTHONHASHSEED')==str(args.seed)
    sys.dont_write_bytecode=True
    firewall=Firewall(args.repo,args.source);sys.addaudithook(firewall)
    cfg,hashes=verify_contract(args.repo,args.source)
    import numpy as np
    import tensorflow as tf
    import tensorflow.contrib.layers
    from tensorflow.core.protobuf import rewriter_config_pb2
    result={'stage':args.stage,'seed':args.seed,'source_pin':PIN,'source_hashes':hashes,'precision_env':dict(PRECISION),'patches':[], 'scientific_execution':False}
    result['runtime']={'python':sys.version,'tensorflow':tf.__version__,'numpy':np.__version__,'tf_build':getattr(tf.sysconfig,'get_build_info',lambda: dict(vars(importlib.import_module('tensorflow.python.platform.build_info'))).get('build_info',{}))()}
    for name in REQUIRED_APIS:
        obj=tf
        for part in name.split('.'):obj=getattr(obj,part)
    assert not tf.executing_eagerly()
    result['A']={'required_apis':list(REQUIRED_APIS),'contrib_unchanged':True,'graph_mode':True}
    def emit():
        result['firewall']={'denied':firewall.denied,'benchmark_images_decoded':0,'test_data_accessed':False}
        print('E03_RESULT='+json.dumps(result,sort_keys=True),flush=True)
    if args.stage=='A':emit();return
    sys.path.insert(0,str(Path(args.source).resolve()))
    before=len(tf.get_default_graph().get_operations())
    train=importlib.import_module('train'); model=importlib.import_module('model.model')
    warp=importlib.import_module('model.warp'); settings=importlib.import_module('model.config').Config
    assert len(tf.get_default_graph().get_operations())==before
    result['B']={'source_imported':True,'graph_ops_created':0,'config_constructor_called':False,'dataset_constructor_called':False}
    if args.stage=='B':emit();return
    # Class attributes are the effective source constants. Avoid filesystem-writing Config.__init__.
    config=settings.__new__(settings)
    tf.set_random_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    images=tf.placeholder(tf.float32,[2,2,256,256,3],name='qualification_images')
    offsets=tf.placeholder(tf.float32,[2,256,256,3],name='qualification_offsets')
    # Diagnostic observers call the original scientific functions unchanged.
    captures={'generators':[],'discriminators':[],'optimizers':[]}
    original_gen,original_disc,original_opt=train.Gen,train.Disc_s,train.get_train_op
    def gen(*a,**kw):
        outputs=original_gen(*a,**kw);captures['generators'].append({'input':a[0].shape.as_list(),'outputs':[x.shape.as_list() for x in outputs]});return outputs
    def disc(*a,**kw):
        outputs=original_disc(*a,**kw);captures['discriminators'].append({'scope':kw['scope'],'input':a[0].shape.as_list(),'outputs':[x.shape.as_list() for x in outputs]});return outputs
    def opt(*a,**kw):
        captures['optimizers'].append({'scope':a[3],'objective':a[0].name});return original_opt(*a,**kw)
    train.Gen,train.Disc_s,train.get_train_op=gen,disc,opt
    try:losses,gop,dop,figure=train._step(config,SimpleNamespace(nextit=(images,offsets)),True)
    finally:train.Gen,train.Disc_s,train.get_train_op=original_gen,original_disc,original_opt
    tv=tf.trainable_variables();actual={v.name:v.shape.as_list() for v in tv}
    assert actual==expected_variables(),{'missing':sorted(set(expected_variables())-set(actual)),'extra':sorted(set(actual)-set(expected_variables()))}
    assert len(tv)==170
    assert all(v.dtype.base_dtype==tf.float32 for v in tv)
    assert [x['input'][1] for x in captures['discriminators']]==[256,160,40]
    assert captures['generators'][0]['outputs']==[[4,32,32,1],[4,1,1,3],[4,1,1,3],[4,64,64,3],[4,256,256,3]]
    ops=tf.get_default_graph().get_operations()
    adam=[o for o in ops if o.type in ['ApplyAdam','ResourceApplyAdam']]
    assert len(adam)==len(tv)
    bn_ops=[o for o in ops if o.type in ['FusedBatchNorm','FusedBatchNormV2','FusedBatchNormV3']]
    # nn_impl.fused_batch_norm clamps the source's 1e-5 to 1.001e-5.
    # This same cuDNN minimum is present in upstream TF 1.13.1 (nn_impl.py:1164).
    assert bn_ops and all(abs(o.get_attr('epsilon')-1.001e-5)<1e-10 for o in bn_ops)
    assert all(o.outputs[0].dtype==tf.float32 for o in bn_ops)
    # Read original kernel inputs, not a separately constructed optimizer oracle.
    beta_inputs=[[o.inputs[i] for i in [6,7,8]] for o in adam]
    gradients=[o.inputs[9] for o in adam]
    gradient_checks=[tf.reduce_all(tf.is_finite(g)) for g in gradients]
    grad_norm=tf.global_norm(gradients)
    global_step=tf.train.get_global_step()
    moving=[v for v in tf.global_variables() if '/moving_' in v.name]
    ema=[v for v in tf.global_variables() if 'ExponentialMovingAverage' in v.name]
    loss_ema=[v for v in tf.global_variables() if v.name.endswith('/avg:0')]
    assert len(moving)==60 and len(ema)==170 and len(loss_ema)==2
    assert tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)==[]
    # updates_collections=None embeds BN updates; preserve source behavior.
    session_config=tf.ConfigProto(allow_soft_placement=False,intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
    session_config.gpu_options.allow_growth=True
    session_config.graph_options.optimizer_options.global_jit_level=tf.OptimizerOptions.OFF
    rewrite=session_config.graph_options.rewrite_options
    rewrite.disable_meta_optimizer=True
    rewrite.auto_mixed_precision=rewriter_config_pb2.RewriterConfig.OFF
    result['session_config']=str(session_config)
    structure={'trainable_variables':actual,'global_variables':{v.name:v.shape.as_list() for v in tf.global_variables()},'captures':captures,'adam_ops':len(adam),'weight_ema_count':len(ema),'loss_ema_count':len(loss_ema),'batch_norm_moving_count':len(moving),'regularization_losses':0,'fused_batch_norm_count':len(bn_ops),'batch_norm_source_epsilon':1e-5,'batch_norm_kernel_epsilon':float(bn_ops[0].get_attr('epsilon')),'parameter_dtype':'float32','update_collection_size':len(tf.get_collection(tf.GraphKeys.UPDATE_OPS))}
    graph_topology=[{'name':o.name,'type':o.type,'inputs':[x.name for x in o.inputs],'controls':[x.name for x in o.control_inputs],'outputs':[{'shape':x.shape.as_list() if x.shape.ndims is not None else None,'dtype':x.dtype.name} for x in o.outputs]} for o in tf.get_default_graph().get_operations()]
    result['graph_topology_sha256']=digest(canonical(graph_topology))
    result['C']={**structure,'structure_sha256':digest(canonical(structure)),'trainable_parameters':int(sum(np.prod(v.shape.as_list()) for v in tv)),'graph_op_count':len(tf.get_default_graph().get_operations())}
    if args.stage=='C':emit();return
    # Synthetic landmarks on an ellipse; no cached geometry or images.
    theta=np.linspace(0,2*np.pi,68,endpoint=False)
    landmarks=np.stack([0.5+0.25*np.cos(theta),0.5+0.30*np.sin(theta)],axis=1).astype(np.float32)
    offset=warp.generate_offset_map(landmarks,landmarks).astype(np.float32)
    assert np.isfinite(offset).all()
    rng=np.random.RandomState(args.seed)
    batch=rng.uniform(0.0,1.0,(2,2,256,256,3)).astype(np.float32)
    feed={images:batch,offsets:np.stack([offset,offset])}
    with tf.Session(config=session_config) as sess:
        devices=[{'name':d.name,'type':d.device_type} for d in sess.list_devices()]
        assert any(d['type']=='GPU' for d in devices)
        sess.run(tf.global_variables_initializer())
        betas=sess.run(beta_inputs)
        assert all(np.allclose(x,[0.9,0.999,1e-8],rtol=1e-6,atol=1e-12) for x in betas)
        lr=float(sess.run(adam[0].inputs[5]));assert np.isclose(lr,6e-5,rtol=1e-6)
        before_step=int(sess.run(global_step))
        initial=digest(b''.join(x.tobytes() for x in sess.run(tv)))
        metadata=tf.RunMetadata()
        values=sess.run(losses,feed_dict=feed,options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=metadata)
        assert all(np.isfinite(v) for v in values) and int(sess.run(global_step))==before_step==0
        gpu_nodes=sum(len(d.node_stats) for d in metadata.step_stats.dev_stats if 'GPU' in d.device.upper())
        assert gpu_nodes>0
        result['D']={'losses':[float(v) for v in values],'finite':True,'gpu_executed_nodes':gpu_nodes,'devices':devices,'input_shape':list(batch.shape),'landmarks_shape':list(landmarks.shape),'initial_weights_sha256':initial,'global_step':before_step,'batch_norm_updates':'Source training-mode forward retains inline moving-stat updates; no optimizer applied.'}
        result['optimizer_validation']={'beta1':float(betas[0][0]),'beta2':float(betas[0][1]),'epsilon':float(betas[0][2]),'initial_learning_rate':lr,'adam_ops':len(adam),'generator_trainable_count':98,'discriminator_trainable_count':72}
        if args.stage=='E' and args.optimizer_path=='G':
            finite,norm=sess.run([gradient_checks,grad_norm],feed_dict=feed)
            assert all(finite) and np.isfinite(norm)
            weights_before=sess.run(tv)
            sess.run(gop,feed_dict=feed)  # EXACTLY ONE Adam optimizer application; official G-only branch.
            after=int(sess.run(global_step));assert after==before_step+1
            weights_after=sess.run(tv)
            changed=[v.name for v,x,y in zip(tv,weights_before,weights_after) if not np.array_equal(x,y)]
            assert changed and all(n.startswith('STDN/') for n in changed)
            assert all(np.isfinite(x).all() for x in weights_after)
            result['E']={'optimizer_applications':1,'branch':'official G-only schedule branch','global_step_before':before_step,'global_step_after':after,'all_G_and_D_gradients_finite':True,'gradient_count':len(finite),'gradient_global_norm':float(norm),'changed_trainable_names':changed,'all_weights_finite':True,'discriminator_update_executed':False,'post_weights_sha256':digest(b''.join(x.tobytes() for x in weights_after))}
        if args.stage=='E' and args.optimizer_path=='D':
            d_adam=[o for o in adam if o.inputs[0].name.startswith('Disc/')]
            assert len(d_adam)==72
            d_loss,d_gradients=sess.run([losses[1],[o.inputs[9] for o in d_adam]],feed_dict=feed)
            assert np.isfinite(d_loss) and all(np.isfinite(x).all() for x in d_gradients)
            d_constants=sess.run([[o.inputs[i] for i in [5,6,7,8]] for o in d_adam])
            assert all(np.allclose(x,[6e-5,0.9,0.999,1e-8],rtol=1e-6,atol=1e-12) for x in d_constants)
            gv=tf.global_variables();state_before=sess.run(gv);weights_before=sess.run(tv)
            graph_before=digest(tf.get_default_graph().as_graph_def().SerializeToString(deterministic=True))
            step_before=int(sess.run(global_step));assert step_before==0
            sess.run(dop,feed_dict=feed)  # EXACTLY ONE original D application; no G application.
            step_after=int(sess.run(global_step));assert step_after==step_before+1
            state_after=sess.run(gv);weights_after=sess.run(tv)
            graph_after=digest(tf.get_default_graph().as_graph_def().SerializeToString(deterministic=True))
            changed=[v.name for v,x,y in zip(tv,weights_before,weights_after) if not np.array_equal(x,y)]
            assert changed and all(n.startswith('Disc/') for n in changed)
            assert all(np.isfinite(x).all() for x in state_after)
            assert graph_before==graph_after
            g_nontrain_changed=[v.name for v,x,y in zip(gv,state_before,state_after)
                                if v.name.startswith('STDN/') and v.name not in actual and not np.array_equal(x,y)]
            assert all('/moving_mean:' in n or '/moving_variance:' in n for n in g_nontrain_changed)
            def subset_hash(prefix,values):
                return digest(b''.join(x.tobytes() for v,x in zip(tv,values) if v.name.startswith(prefix)))
            normalized=sorted([dict(o,controls=sorted(o['controls'])) for o in graph_topology],key=lambda o:o['name'])
            result['D_optimizer']={'optimizer_applications':1,'generator_optimizer_applications':0,
                'loss_before_application':float(d_loss),'loss_finite':True,'gradient_count':len(d_gradients),
                'all_gradients_finite':True,'gradient_global_norm':float(np.sqrt(sum(np.sum(x.astype(np.float64)**2) for x in d_gradients))),
                'completed':True,'changed_discriminator_trainable_names':changed,'discriminator_trainable_count':72,
                'generator_trainable_count':98,'generator_trainable_variables_unchanged':True,
                'generator_nontrainable_changed_names':g_nontrain_changed,
                'generator_state_note':'Original training-mode forward embeds BN moving-stat updates. They are preserved, not optimizer updates to G weights.',
                'global_step_before':step_before,'global_step_after':step_after,
                'source_global_step_policy':'Both original get_train_op calls use opt.apply_gradients(grads, global_step=global_step); each application increments the shared step.',
                'all_global_variables_finite':True,'graph_before_sha256':graph_before,'graph_after_sha256':graph_after,
                'canonical_topology_sha256':digest(canonical(normalized)),
                'adam_kernel_inputs_all_72':[[float(y) for y in x] for x in d_constants],
                'generator_before_sha256':subset_hash('STDN/',weights_before),'generator_after_sha256':subset_hash('STDN/',weights_after),
                'discriminator_before_sha256':subset_hash('Disc/',weights_before),'discriminator_after_sha256':subset_hash('Disc/',weights_after),
                'post_trainable_weights_sha256':digest(b''.join(x.tobytes() for x in weights_after))}
        # Record libraries actually mapped; no system nvcc inference.
        result['loaded_gpu_libraries']=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if any(n in line for n in ['libcudart','libcudnn','libcublas','libcuda.so'])})
    emit()


if __name__=='__main__':
    main()
