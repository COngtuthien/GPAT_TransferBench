"""Individual paper losses and frozen M6D3d weighted compositions."""


def squared_norm(tf, x):
    """Batch expectation of squared Frobenius norm; no hidden spatial averaging."""
    return tf.reduce_mean(tf.reduce_sum(tf.square(x), axis=list(range(1, x.shape.ndims))))


def absolute_norm(tf, x):
    return tf.reduce_mean(tf.reduce_sum(tf.abs(x), axis=list(range(1, x.shape.ndims))))


def mask_loss(tf, mask, additive, beta):
    # Broadcast single-channel P against the RGB comparison as written. No
    # unrequested absolute-value/channel-collapse operator is introduced.
    target = tf.cast(additive > beta, tf.float32, name='mask_positive_target')
    p0 = tf.zeros_like(mask, name='P0')
    negative = tf.constant(0., tf.float32, name='P0_negative_prior')
    positive = squared_norm(tf, mask - target)
    return positive + negative, p0, negative, positive


def objectives(tf, original, hard, depth_target, predictions, target_trace):
    result = {}
    result['L_depth'] = absolute_norm(tf, original['depth'] - depth_target) / (32*32)
    result['L_G'] = tf.add_n([squared_norm(tf, p[4:]-1.) for p in predictions])
    result['L_D'] = tf.add_n([squared_norm(tf, p[:4]-1.) + squared_norm(tf, p[4:]) for p in predictions])
    result['L_P'], result['P0'], result['P0_negative_prior'], result['P_positive'] = mask_loss(
        tf, original['P'][4:], original['T_A'][4:], .1)
    result['L_R'] = tf.add_n([squared_norm(tf, original[k]) for k in ['B', 'C', 'T', 'P']])
    result['L_S'] = absolute_norm(tf, hard['trace'][4:] - target_trace)
    result['L_H'] = absolute_norm(tf, hard['depth'][4:]) / (32*32)
    result['loss_G_step1'] = 100*result['L_depth'] + 5*result['L_G'] + result['L_P'] + 1e-4*result['L_R']
    result['loss_D'] = result['L_D']
    result['loss_G_step3'] = 10*result['L_S'] + result['L_H']
    return result
