"""M6D3d PhySTD topology. Caller supplies TensorFlow 1; no data loading."""


def resize(tf, x, size, name):
    # Native TF1 bilinear coordinates, no align-corners or antialias substitution.
    # CPU placement supports deterministic native resize gradients in TF1.
    with tf.device('/cpu:0'), tf.name_scope(name):
        return tf.image.resize_bilinear(x, [size, size], align_corners=False,
                                        half_pixel_centers=False)


def lowpass(tf, x, size, name):
    with tf.name_scope(name):
        return resize(tf, resize(tf, x, size, 'down'), 256, 'up')


def frequency_input(tf, x):
    low32 = lowpass(tf, x, 32, 'input_low32')
    low128 = lowpass(tf, x, 128, 'input_low128')
    bands = {'I_B': low32, 'I_C': low128 - low32, 'I_T': x - low128}
    bands['frequency_input'] = tf.concat([bands['I_B'], 15 * bands['I_C'],
                                        25 * bands['I_T']], 3, name='frequency_input')
    return bands


class Layers:
    def __init__(self, tf):
        self.tf = tf
        self.initializers = {}

    def conv(self, x, channels, name, stride=1, kernel=3, transpose=False, output=False):
        tf = self.tf
        with tf.variable_scope(name):
            cin = x.shape.as_list()[-1]
            shape = [kernel, kernel, channels, cin] if transpose else [kernel, kernel, cin, channels]
            w = tf.get_variable('kernel', shape, dtype=tf.float32,
                                initializer=tf.random_normal_initializer(mean=0., stddev=.02))
            b = tf.get_variable('bias', [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
            self.initializers[w.name] = {'distribution': 'Normal', 'mean': 0., 'stddev': .02}
            self.initializers[b.name] = {'distribution': 'constant', 'value': 0., 'role': 'bias'}
            if transpose:
                static = x.shape.as_list()
                target = [static[0], static[1] * stride, static[2] * stride, channels]
                y = tf.nn.conv2d_transpose(x, w, target, [1, stride, stride, 1], padding='SAME')
                y.set_shape(target)
            else:
                y = tf.nn.conv2d(x, w, [1, stride, stride, 1], padding='SAME')
            y = tf.nn.bias_add(y, b)
            if not output:
                y = tf.layers.batch_normalization(y, momentum=.99, epsilon=1e-5,
                    center=True, scale=True, training=True, fused=True, name='BN')
                scope = tf.get_variable_scope().name
                self.initializers[scope + '/BN/gamma:0'] = {'distribution': 'constant', 'value': 1., 'role': 'BN scale'}
                self.initializers[scope + '/BN/beta:0'] = {'distribution': 'constant', 'value': 0., 'role': 'BN offset'}
                y = tf.nn.leaky_relu(y, alpha=.2, name='leaky_relu')
            return y

    def attention(self, x, kernel, name):
        tf = self.tf
        with tf.variable_scope(name):
            stats = tf.concat([tf.reduce_max(x, 3, keepdims=True),
                               tf.reduce_mean(x, 3, keepdims=True)], 3)
            gate = tf.sigmoid(self.conv(stats, 1, 'projection', kernel=kernel, output=True))
            return x * gate, gate


def generator(tf, x, layers):
    result = frequency_input(tf, x)
    with tf.variable_scope('G', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('encoder'):
            z = layers.conv(result['frequency_input'], 32, 'input')
            for i, width in enumerate([64, 96, 128], 1):
                # Fig.4 ConvBlock inset: three convolutions, final downsampling.
                with tf.variable_scope('block%d' % i):
                    z = layers.conv(z, width, 'conv1')
                    z = layers.conv(z, width, 'conv2')
                    z = layers.conv(z, width, 'down', stride=2)
                result['F%d' % i] = z
        with tf.variable_scope('decoder'):
            z = layers.conv(z, 64, 'entry', transpose=True)
            for i, skip in enumerate(['F3', 'F2', 'F1'], 1):
                z = tf.concat([z, result[skip]], 3, name='shortcut%d' % i)
                z = layers.conv(z, 32, 'U%d' % i, stride=2, transpose=True)
                result['U%d' % i] = z
            raw = layers.conv(z, 13, 'output', output=True)
            result['decoder_raw'] = raw
            for name, start, end in [('B', 0, 3), ('C', 3, 6), ('T', 6, 9), ('I_P', 10, 13)]:
                result[name] = tf.identity(raw[..., start:end], name=name)
            result['P'] = tf.sigmoid(raw[..., 9:10], name='P')
        with tf.variable_scope('depth'):
            parts = []
            for key, kernel in [('F1', 7), ('F2', 5), ('F3', 3), ('U3', 7)]:
                attended, gate = layers.attention(result[key], kernel, 'attention_' + key)
                result['attention_' + key] = gate
                parts.append(resize(tf, attended, 32, 'resize_' + key))
            z = layers.conv(tf.concat(parts, 3), 32, 'conv')
            result['depth'] = tf.sigmoid(layers.conv(z, 1, 'output', output=True), name='depth')
    result['B_low'] = lowpass(tf, result['B'], 32, 'B_low32')
    result['C_low'] = lowpass(tf, result['C'], 128, 'C_low128')
    result['T_A'] = tf.add_n([result['B_low'], result['C_low'], result['T']], name='T_A')
    result['reconstruction'] = (1 - result['P']) * (x - result['T_A']) + result['P'] * result['I_P']
    result['trace'] = x - result['reconstruction']
    return result


def discriminator(tf, x, layers, index, size):
    with tf.variable_scope('D/D%d' % index, reuse=tf.AUTO_REUSE):
        z = resize(tf, x, size, 'input_resize')
        for i, (width, stride) in enumerate([(32, 2), (64, 2), (96, 2), (128, 1)], 1):
            z = layers.conv(z, width, 'conv%d' % i, stride=stride)
        return tf.sigmoid(layers.conv(z, 1, 'output', output=True), name='probability')
