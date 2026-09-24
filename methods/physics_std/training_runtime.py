"""Clean-process synthetic qualification, not a benchmark training runner.

Run directly with the read-only E03 interpreter. Imports the graph as an isolated
package to avoid importing the Python-3.12 geometry/controller dependency tree.
Only hashes and scalar diagnostics are persisted; never model tensors.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True
SEED = 314159
ACTIVE_REPORT = {'optimizer_applications': 0}
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


class ScientificGateError(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise ScientificGateError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def forbidden(path):
    p = Path(os.path.abspath(os.fsdecode(path)))
    parts = p.parts
    data_component = any(s in parts for s in ('faces_256', 'manifests'))
    runtime_data = any(part.endswith('_runtime') and i+1 < len(parts) and
                       parts[i+1] in ('data', 'runs', 'banks', 'cache')
                       for i,part in enumerate(parts))
    repo_data = any(part == 'GPAT_TransferBench' and i+1 < len(parts) and
                    parts[i+1] in ('data', 'cache', 'runs', 'manifests')
                    for i,part in enumerate(parts))
    return data_component or runtime_data or repo_data or p.suffix.lower() in {
        '.parquet','.jpg','.jpeg','.png','.bmp','.webp','.avi','.mp4','.npy','.npz',
        '.ckpt','.pth','.pt','.pkl','.h5'}


class Firewall:
    def __init__(self):
        self.denied = []
        self.events = {'open':0, 'os.listdir':0, 'os.scandir':0}

    def __call__(self, event, args):
        if event not in self.events or not args or not isinstance(args[0], (str, bytes)):
            return
        self.events[event] += 1
        path = os.path.abspath(os.fsdecode(args[0]))
        deny = forbidden(path)
        if event == 'open':
            flags = args[2] if len(args)>2 and isinstance(args[2],int) else 0
            writing = flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)
            deny = deny or (writing and not path.startswith('/tmp/') and path != '/dev/null')
        if deny:
            self.denied.append({'event':event,'path':path})
            raise ScientificGateError('DATA/WRITE FIREWALL: ' + event + ' ' + path)


def import_graph_package():
    package = types.ModuleType('_e04_tensor_graph')
    package.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules[package.__name__] = package
    import importlib
    return (importlib.import_module('_e04_tensor_graph.training_graph'),
            importlib.import_module('_e04_tensor_graph.trace_transfer'))


def fixtures(transfer):
    import numpy as np
    y,x = np.mgrid[:256,:256].astype(np.float32)
    rgb = []
    for i in range(8):
        z = np.stack([.5+.18*np.sin(x/(12+i))+.12*np.cos(y/17),
            .5+.2*np.cos((x+y)/(20+i)), .5+.15*np.sin(y/(9+i))],-1)
        if i >= 4:
            z += .04*np.sin(x[...,None]*1.7+y[...,None]*.6)
        rgb.append(z)
    rgb = np.asarray(rgb,dtype=np.float32)
    dy,dx = np.mgrid[:32,:32].astype(np.float32)
    depth = np.zeros((8,32,32,1),np.float32)
    for i in range(4):
        depth[i,...,0] = .8*np.sqrt(np.maximum(0.,1-((dx-15.5)/(10+i*.3))**2-((dy-15.5)/13)**2))
    tx,ty = np.meshgrid(np.linspace(0,255,14),np.linspace(0,255,10))
    target = np.stack([tx,ty],-1).reshape(140,2).astype(np.float32)
    offsets = []
    for i in range(4):
        source = target.copy()
        source[:,0] += (.4+i*.1)*np.sin(target[:,1]/37)
        source[:,1] += .3*np.cos(target[:,0]/41)
        offsets.append(transfer.dense_offsets(source,target))
    rng = np.random.RandomState(SEED)
    scales = rng.uniform(0,1,(4,1,1,3)).astype(np.float32)
    keep = (rng.uniform(0,1,(4,1,1,1)) >= .5).astype(np.float32)
    return rgb,depth,np.asarray(offsets),scales,keep


def summarize(value):
    import numpy as np
    a = np.asarray(value)
    require(np.isfinite(a).all(),'nonfinite diagnostic tensor')
    return {'shape':list(a.shape),'dtype':str(a.dtype),'sha256':digest(a.tobytes()),
            'min':float(a.min()),'max':float(a.max()),
            'mean':float(a.mean(dtype=np.float64)),
            'l2':float(np.linalg.norm(a.astype(np.float64).ravel()))}


def expected_connection(loss, name):
    if loss == 'L_D':
        return name.startswith('D/')
    if loss in ('L_depth','L_H'):
        return name.startswith('G/') and not name.startswith('G/decoder/output/')
    return name.startswith('G/encoder/') or name.startswith('G/decoder/')


def numpy_resize(x, size):
    """Independent native-TF1 bilinear coordinate oracle, no image library."""
    import numpy as np
    h,w = x.shape[1:3]
    yy = np.arange(size,dtype=np.float32)*np.float32(h/size)
    xx = np.arange(size,dtype=np.float32)*np.float32(w/size)
    y0,x0 = np.floor(yy).astype(int),np.floor(xx).astype(int)
    y1,x1 = np.minimum(y0+1,h-1),np.minimum(x0+1,w-1)
    dx,dy = (xx-x0).astype(np.float32)[None,None,:,None],(yy-y0).astype(np.float32)[None,:,None,None]
    a,b = x[:,y0[:,None],x0[None,:]],x[:,y0[:,None],x1[None,:]]
    c,d = x[:,y1[:,None],x0[None,:]],x[:,y1[:,None],x1[None,:]]
    top,bottom = a+(b-a)*dx,c+(d-c)*dx
    return top+(bottom-top)*dy


def weighted_check(values):
    import numpy as np
    a = {k:np.float32(v) for k,v in values.items()}
    expected1 = ((np.float32(100)*a['L_depth']+np.float32(5)*a['L_G'])+a['L_P'])+np.float32(1e-4)*a['L_R']
    expected3 = np.float32(10)*a['L_S']+a['L_H']
    require(expected1 == a['loss_G_step1'] and expected3 == a['loss_G_step3'],
            'outside-graph FP32 weighted sum mismatch')
    return {'step1_numpy_fp32':float(expected1),'step3_numpy_fp32':float(expected3),
            'step1_absolute_error':0.,'step3_absolute_error':0.,'exact':True}


def qualify(root):
    import numpy as np
    import tensorflow as tf
    graph_module, transfer = import_graph_package()
    graph = tf.Graph()
    with graph.as_default(), tf.device('/gpu:0'):
        tf.set_random_seed(SEED)
        model = graph_module.TrainingGraph(tf,root)
        # Add diagnostics before fingerprinting, without optimizer applications.
        ip_gradient = tf.gradients(model.losses['L_G'],model.original['I_P'])[0]
        p_gradient = tf.gradients(model.losses['L_P'],model.original['P'])[0]
        require(ip_gradient is not None and p_gradient is not None,'inpainting disconnected')
        stopped = tf.gradients(tf.reduce_sum(model.target_trace)+tf.reduce_sum(model.hard_input),model.G)
        require(all(g is None for g in stopped),'supervision stop-gradient failure')
        lr_probe = tf.placeholder(tf.int64,[],name='lr_probe_iteration')
        probe_rates = graph_module.learning_rates(tf,lr_probe)
        variables = tf.global_variables()
        trainables = tf.trainable_variables()
        init = tf.global_variables_initializer()
    require(set(v.name for v in trainables) == set(v.name for v in model.G+model.D),'unexpected trainable group')
    require(len(trainables) == len(model.G)+len(model.D),'overlapping trainable groups')
    require(model.slot_references[1] == model.slot_references[3],'G slots differ')
    with graph.as_default():
        gp = model.opt_G._get_beta_accumulators()
        dp = model.opt_D._get_beta_accumulators()
    require(not set(v.name for v in gp)&set(v.name for v in dp),'shared G/D beta state')
    cfg = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
    cfg.gpu_options.allow_growth = True
    cfg.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.OFF
    cfg.graph_options.rewrite_options.disable_meta_optimizer = True
    cfg.graph_options.rewrite_options.auto_mixed_precision = 2
    # NVIDIA TF1 has no separate MKL AMP field; disable it only if exposed.
    if 'auto_mixed_precision_mkl' in cfg.graph_options.rewrite_options.DESCRIPTOR.fields_by_name:
        cfg.graph_options.rewrite_options.auto_mixed_precision_mkl = 2
    inputs = fixtures(transfer)
    feed = dict(zip([model.rgb,model.depth,model.offsets,model.hard_scales,model.hard_keep],inputs))
    require(inputs[0].shape == (8,256,256,3),'synthetic RGB shape')
    require(np.count_nonzero(inputs[1][4:]) == 0,'spoof fixture depth must be zero')
    trainable_names = set(v.name for v in trainables)
    inventory = [{'name':v.name,'shape':v.shape.as_list(),'dtype':v.dtype.base_dtype.name,
                  'trainable':v.name in trainable_names} for v in variables]
    counts = {}
    for v in trainables:
        component = '/'.join(v.name.split('/')[:2])
        counts[component] = counts.get(component,0)+int(np.prod(v.shape.as_list()))
    report = {'qualification_seed':SEED,'tensorflow':tf.__version__,'precision_env':dict(PRECISION),
        'session_config':str(cfg),'variables':inventory,'parameters':counts,
        'total_trainable_parameters':sum(counts.values()),'operation_count':len(graph.get_operations()),
        'graph_def_sha256':digest(graph.as_graph_def().SerializeToString(deterministic=True)),
        'operation_structure_sha256':digest(json.dumps([(o.name,o.type,[t.name for t in o.inputs],
                [c.name for c in o.control_inputs]) for o in graph.get_operations()],separators=(',',':')).encode()),
        'groups':{'G':[v.name for v in model.G],'D':[v.name for v in model.D]},
        'optimizer_state':{'step_slot_references':model.slot_references,
            'G_beta_powers':[v.name for v in gp],'D_beta_powers':[v.name for v in dp],
            'generator_instances':1,'discriminator_instances':1,'shared_G_slots':True},
        'outputs':{k:{'name':v.name,'shape':v.shape.as_list()} for k,v in model.outputs.items()},
        'fixture_sha256':[digest(x.tobytes()) for x in inputs],
        'fixture_spec':{'RGB':[8,256,256,3],'depth':[8,32,32,1],'live':4,'spoof':4,
                        'spoof_depth_exact_zero':True,'hard_depth_exact_zero':True,'geometry_correspondences':140},
        'stop_gradient_verified':True,'optimizer_applications':0,'steps':[]}
    global ACTIVE_REPORT
    ACTIVE_REPORT = report
    scalar_keys = ['L_depth','L_G','L_P','L_R','L_S','L_H','L_D','loss_G_step1','loss_D','loss_G_step3','P0_negative_prior','P_positive']
    scalar_fetch = {k:model.losses[k] for k in scalar_keys}
    def snapshot(session):
        return {v.name:summarize(a) for v,a in zip(variables,session.run(variables))}
    with tf.Session(graph=graph,config=cfg) as session:
        session.run(init)
        report['devices'] = [{'name':d.name,'type':d.device_type} for d in session.list_devices()]
        require(any(d['type']=='GPU' for d in report['devices']),'GPU not in session')
        initial_values = session.run(trainables)
        report['initialization'] = {}
        for v,a in zip(trainables,initial_values):
            spec = model.layers.initializers[v.name]
            record = dict(spec, measured=summarize(a), stddev_measured=float(a.std()),initializer=v.initial_value.name)
            if v.name.endswith('/kernel:0'):
                pending,seen = [v.initial_value.op],set()
                while pending:
                    op = pending.pop()
                    if op in seen: continue
                    seen.add(op); pending.extend(t.op for t in op.inputs)
                require(any(o.type=='RandomStandardNormal' for o in seen),'kernel not native normal: '+v.name)
                require(spec == {'distribution':'Normal','mean':0.,'stddev':.02},'kernel initializer parameters')
            else:
                require(np.all(a == spec['value']),'constant initializer '+v.name)
            report['initialization'][v.name] = record
        report['initial_state'] = snapshot(session)
        metadata = tf.RunMetadata()
        output_values = session.run(model.outputs,feed,
            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=metadata)
        placements = {node.node_name:device.device for device in metadata.step_stats.dev_stats
                      for node in device.node_stats}
        report['convolution_placements'] = {op.name:placements.get(op.name) for op in graph.get_operations()
                                           if op.type in ('Conv2D','Conv2DBackpropInput') and op.name in placements}
        require(any('gpu' in (p or '').lower() for p in report['convolution_placements'].values()),'no GPU convolution executed')
        report['forward'] = {k:summarize(v) for k,v in output_values.items()}
        low32 = numpy_resize(numpy_resize(inputs[0],32),256)
        low128 = numpy_resize(numpy_resize(inputs[0],128),256)
        oracle = {'I_B':low32,'I_C':low128-low32,'I_T':inputs[0]-low128}
        oracle['frequency_input'] = np.concatenate([oracle['I_B'],15*oracle['I_C'],25*oracle['I_T']],3)
        report['frequency_oracle_max_errors'] = {k:float(np.max(np.abs(v-output_values['original/'+k]))) for k,v in oracle.items()}
        require(max(report['frequency_oracle_max_errors'].values())<=4e-6,'frequency decomposition oracle mismatch')
        for key,shape in [('F1',[8,128,128,64]),('F2',[8,64,64,96]),('F3',[8,32,32,128]),
                           ('decoder_raw',[8,256,256,13]),('depth',[8,32,32,1])]:
            require(list(output_values['original/'+key].shape)==shape,'resolved shape '+key)
        require(np.any(output_values['original/P'] != 0),'P is forced zero')
        require(np.all(output_values['P0']==0),'P0 is not exact zero')
        require(output_values['original/depth'].min()>=0 and output_values['original/depth'].max()<=1,'depth range')
        ipg,pg = session.run([ip_gradient,p_gradient],feed)
        require(np.any(ipg != 0) and np.any(pg != 0),'active inpainting gradients are zero')
        report['inpainting'] = {'I_P_adversarial_gradient':summarize(ipg),'P_primary_gradient':summarize(pg),
            'P0_exact_zero':True,'negative_prior_op_type':model.losses['P0_negative_prior'].op.type,
            'negative_prior_inputs':len(model.losses['P0_negative_prior'].op.inputs)}
        require(model.losses['P0_negative_prior'].op.type=='Const','P0 quotient evaluated')
        values = session.run(scalar_fetch,feed)
        report['initial_losses'] = {k:float(v) for k,v in values.items()}
        require(all(np.isfinite(v) for v in values.values()),'nonfinite initial loss')
        report['weighted_initial'] = weighted_check(values)
        report['gradient_inventory'] = {}
        for loss,pairs in model.per_loss_gradients.items():
            for gradient,variable in pairs:
                require((gradient is not None)==expected_connection(loss,variable.name),
                        'unexpected loss connectivity: '+loss+' -> '+variable.name)
            active = [(g,v) for g,v in pairs if g is not None]
            evaluated = session.run([g for g,_ in active],feed)
            record = {v.name:summarize(g) for (_,v),g in zip(active,evaluated)}
            require(any(r['l2']>0 for r in record.values()),'loss has zero gradient '+loss)
            report['gradient_inventory'][loss] = {'connected':record,
                'intentionally_disconnected':[v.name for g,v in pairs if g is None]}
        report['lr_boundaries'] = []
        for t,expected in [(0,5e-5),(44999,5e-5),(45000,5e-6),(89999,5e-6),(90000,5e-7),(135000,5e-8)]:
            g,d = session.run(probe_rates,{lr_probe:t})
            require(np.isclose(g,expected,rtol=2e-7,atol=0) and d == g*.5,'LR boundary')
            report['lr_boundaries'].append({'iteration':t,'G':float(g),'D':float(d),'expected_G':expected,'expected_D':expected*.5})
        require(snapshot(session)==report['initial_state'],'pure diagnostics changed native state')
        trainable_names = set(v.name for v in trainables)
        for step,key in [(1,'loss_G_step1'),(2,'loss_D'),(3,'loss_G_step3')]:
            before = snapshot(session)
            t0,phase = session.run([model.iteration,model.phase])
            require(int(t0)==0 and int(phase)==step-1,'pre-step iteration/phase')
            values = session.run(scalar_fetch,feed)
            require(all(np.isfinite(v) for v in values.values()),'nonfinite pre-step loss')
            recompute = weighted_check(values)
            pairs = model.step_gradients[step]
            require(all(g is not None for g,_ in pairs),'optimizer objective disconnected variable')
            grads = session.run([g for g,_ in pairs],feed)
            grads_record = {v.name:summarize(g) for (_,v),g in zip(pairs,grads)}
            rates = session.run([model.lr_G,model.lr_D])
            # Serialize native BN assignments, especially the shared two-pass G state.
            bn_changes = []
            for updates in model.bn_paths[step]:
                prev = snapshot(session)
                session.run(updates,feed)
                nxt = snapshot(session)
                changed = [n for n in prev if prev[n]['sha256'] != nxt[n]['sha256']]
                require(not set(changed)&trainable_names,'BN update mutated trainables')
                bn_changes.append(changed)
            session.run(model.apply[step],feed)
            report['optimizer_applications'] += 1
            after = snapshot(session)
            changed = [n for n in before if before[n]['sha256'] != after[n]['sha256']]
            group = model.D if step==2 else model.G
            intended = set(v.name for v in group)
            changed_trainable = set(changed)&trainable_names
            require(bool(changed_trainable) and changed_trainable<=intended,'optimizer trainable isolation')
            require(any(n.endswith('/kernel:0') for n in changed_trainable),'no intended kernel update')
            t1,phase1 = session.run([model.iteration,model.phase])
            require(int(t1)==(1 if step==3 else 0),'iteration must increment once after step3')
            require(int(phase1)==(0 if step==3 else step),'phase progression')
            beta_values = session.run(list(gp)+list(dp))
            updates_G = 2 if step==3 else 1
            updates_D = 0 if step==1 else 1
            expected_beta = [.9**(updates_G+1),.999**(updates_G+1),.9**(updates_D+1),.999**(updates_D+1)]
            require(np.allclose(beta_values,expected_beta,rtol=3e-7,atol=0),'shared Adam beta powers advancement')
            report['steps'].append({'step':step,'objective':key,'losses_before':{k:float(v) for k,v in values.items()},
                'weighted_recomputation':recompute,'gradients':grads_record,'iteration_before':int(t0),
                'iteration_after':int(t1),'phase_after':int(phase1),'lr_G':float(rates[0]),'lr_D':float(rates[1]),
                'changed_trainables':sorted(changed_trainable),'unchanged_intended_trainables':sorted(intended-changed_trainable),
                'unrelated_trainables_unchanged':True,'bn_changes_by_forward_collection':bn_changes,
                'changed_nontrainables':sorted(set(changed)-trainable_names),
                'G_and_D_beta_values':[float(v) for v in beta_values],'post_state':after})
        report['final_iteration'] = int(session.run(model.iteration))
        report['final_losses'] = {k:float(v) for k,v in session.run(scalar_fetch,feed).items()}
        require(all(np.isfinite(v) for v in report['final_losses'].values()),'nonfinite post-minibatch losses')
        import ctypes
        env = Path(sys.executable).parents[1]
        cuda = ctypes.CDLL(str(env/'lib/python3.8/site-packages/nvidia/cuda_runtime/lib/libcudart.so.11.0'))
        cudnn = ctypes.CDLL(str(env/'lib/python3.8/site-packages/nvidia/cudnn/lib/libcudnn.so.8'))
        version = ctypes.c_int()
        rc = cuda.cudaRuntimeGetVersion(ctypes.byref(version))
        cudnn.cudnnGetVersion.restype = ctypes.c_size_t
        report['native_runtime'] = {'cudaRuntimeGetVersion':version.value,'cuda_query_rc':rc,
                                    'cudnnGetVersion':int(cudnn.cudnnGetVersion())}
    report['status'] = 'PASS'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    require(os.path.abspath(args.output).startswith('/tmp/'),'qualification evidence must be outside run directories')
    os.environ.update(PRECISION)
    firewall = Firewall()
    sys.addaudithook(firewall)
    root = Path(__file__).resolve().parents[2]
    try:
        result = qualify(root)
        spec = importlib.util.spec_from_file_location('m6d3e_checks',root/'tests/test_m6d3e_e04_training_graph.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result['tests'] = module.run_evidence_tests(result)
        require(result['tests']['failures']==0 and result['tests']['errors']==0,'qualification tests failed')
    except Exception as exc:
        import traceback
        result = dict(ACTIVE_REPORT, status='STOP_AND_REPORT',error_type=type(exc).__name__,
                      error=str(exc),traceback=traceback.format_exc())
    result['firewall'] = {'events':firewall.events,'denied':firewall.denied,
        'scope':'Python open/listdir/scandir audit, not an OS-wide syscall trace'}
    Path(args.output).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':result['status'],'output':args.output,
                      'optimizer_applications':result.get('optimizer_applications'),
                      'error':result.get('error'),'tests':result.get('tests')}))
    return 0 if result['status']=='PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
