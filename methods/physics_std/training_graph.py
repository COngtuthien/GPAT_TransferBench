"""Resolved three-step PhySTD graph; tensor inputs only, no benchmark runner."""
from .network import Layers, generator, discriminator, lowpass
from .trace_transfer import warp, synthesis
from .losses import objectives
from .training_contract import load_contract


def learning_rates(tf, iteration):
    lr = tf.constant(5e-5, tf.float32) * tf.pow(tf.constant(10., tf.float32),
        -tf.cast(tf.floordiv(iteration, 45000), tf.float32))
    return tf.identity(lr, 'lr_G'), tf.identity(lr*.5, 'lr_D')


class TrainingGraph:
    def __init__(self, tf, root=None):
        self.contract = load_contract(root)
        self.tf = tf
        self.layers = Layers(tf)
        self.rgb = tf.placeholder(tf.float32, [8,256,256,3], name='synthetic_rgb')
        self.depth = tf.placeholder(tf.float32, [8,32,32,1], name='synthetic_depth')
        self.offsets = tf.placeholder(tf.float32, [4,256,256,2], name='sampling_offsets')
        self.hard_scales = tf.placeholder(tf.float32, [4,1,1,3], name='hard_scales')
        self.hard_keep = tf.placeholder(tf.float32, [4,1,1,1], name='hard_keep_inpainting')
        self.iteration = tf.get_variable('global_iteration', [], tf.int64, initializer=tf.zeros_initializer(), trainable=False)
        self.phase = tf.get_variable('minibatch_phase', [], tf.int32, initializer=tf.zeros_initializer(), trainable=False)
        self.lr_G, self.lr_D = learning_rates(tf, self.iteration)
        self.original = generator(tf, self.rgb, self.layers)
        self.bn_original = list(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        live, source = self.rgb[:4], self.rgb[4:]
        g = self.original
        p, content = g['P'][4:], g['I_P'][4:]
        b,c,t = g['B_low'][4:],g['C_low'][4:],g['T'][4:]
        self.hierarchical = [(1-p)*(lowpass(tf,source,32,'source_low32')-b)+p*content,
            (1-p)*(lowpass(tf,source,128,'source_low128')-b-c)+p*content,
            g['reconstruction'][4:]]
        self.warped = {k: warp(tf, value, self.offsets, 'warp_'+k) for k,value in
                       [('B',b),('C',c),('T',t),('P',p),('source',source)]}
        w = self.warped
        self.spoof = synthesis(tf,live,w['source'],w['B']+w['C']+w['T'],w['P'])
        self.predictions = [discriminator(tf,tf.concat([live,fake],0),self.layers,i,size)
            for i,size,fake in zip([1,2,3],[32,96,256],self.hierarchical)]
        self.predictions.append(discriminator(tf,tf.concat([source,self.spoof],0),self.layers,4,256))
        self.bn_D = tf.get_collection(tf.GraphKeys.UPDATE_OPS)[len(self.bn_original):]
        hard_trace = sum(w[key]*self.hard_scales[...,i:i+1] for i,key in enumerate(['B','C','T']))
        hard_mask = w['P']*self.hard_keep
        self.hard_rgb = synthesis(tf,live,w['source'],hard_trace,hard_mask)
        self.target_trace = tf.stop_gradient(self.hard_rgb-live, name='ground_truth_trace_stop')
        self.hard_input = tf.stop_gradient(tf.concat([live,self.hard_rgb],0), name='hard_input_stop')
        self.hard = generator(tf, self.hard_input, self.layers)
        self.bn_hard = tf.get_collection(tf.GraphKeys.UPDATE_OPS)[len(self.bn_original)+len(self.bn_D):]
        self.losses = objectives(tf,self.original,self.hard,self.depth,self.predictions,self.target_trace)
        self.G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G/')
        self.D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D/')
        self.per_loss_gradients = {}
        for key in ['L_depth','L_G','L_P','L_R','L_S','L_H','L_D']:
            group = self.D if key == 'L_D' else self.G
            self.per_loss_gradients[key] = list(zip(tf.gradients(self.losses[key],group),group))
        self.opt_G = tf.train.AdamOptimizer(self.lr_G,beta1=.9,beta2=.999,epsilon=1e-8,name='G_Adam')
        self.opt_D = tf.train.AdamOptimizer(self.lr_D,beta1=.9,beta2=.999,epsilon=1e-8,name='D_Adam')
        self.step_gradients, self.apply, self.slot_references = {}, {}, {}
        for step, key, opt, group in [(1,'loss_G_step1',self.opt_G,self.G),
                                     (2,'loss_D',self.opt_D,self.D),(3,'loss_G_step3',self.opt_G,self.G)]:
            grads = opt.compute_gradients(self.losses[key],var_list=group)
            self.step_gradients[step] = grads
            with tf.control_dependencies([tf.assert_equal(self.phase,step-1,message='G/D/G order')]):
                apply = opt.apply_gradients(grads, name='apply_step%d' % step)
            self.slot_references[step] = [(v.name,opt.get_slot(v,'m').name,opt.get_slot(v,'v').name) for _,v in grads]
            with tf.control_dependencies([apply]):
                complete = tf.assign_add(self.iteration,1) if step == 3 else tf.identity(self.iteration)
            with tf.control_dependencies([complete]):
                self.apply[step] = tf.assign(self.phase, 0 if step == 3 else step, name='finish_step%d' % step)
        self.bn_paths = {1:[self.bn_original,self.bn_D],2:[self.bn_original,self.bn_D],
                         3:[self.bn_original,self.bn_hard]}
        self.outputs = {}
        for prefix, mapping in [('original',self.original),('hard',self.hard),('warped',self.warped)]:
            self.outputs.update({prefix+'/'+k:v for k,v in mapping.items()})
        self.outputs.update({'hierarchical/%d'%i:x for i,x in enumerate(self.hierarchical)})
        self.outputs.update({'D%d'%i:x for i,x in enumerate(self.predictions,1)})
        self.outputs.update({'synthesized_spoof':self.spoof,'hard_rgb':self.hard_rgb,
                             'hard_input':self.hard_input,'target_trace':self.target_trace,
                             'P0':self.losses['P0']})
