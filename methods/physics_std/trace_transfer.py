"""Geometric trace transfer from supplied Q140 coordinates or dense offsets.

No 3DDFA/model/image access. Offsets are inverse sampling coordinates: output
target pixel p reads source p+offset[p]. The 140 rows are correspondence slots
in the frozen A3 order; generated fixtures do not rederive vertex identities.
"""


def dense_offsets(source_xy, target_xy):
    import numpy as np
    from scipy.interpolate import LinearNDInterpolator
    source_xy = np.asarray(source_xy, dtype=np.float32)
    target_xy = np.asarray(target_xy, dtype=np.float32)
    if source_xy.shape != (140, 2) or target_xy.shape != (140, 2):
        raise ValueError('Q140 ordered [140,2] correspondences required')
    if not np.isfinite(source_xy).all() or not np.isfinite(target_xy).all():
        raise ValueError('finite correspondence coordinates required')
    y, x = np.mgrid[:256, :256]
    grid = np.stack([x, y], -1)
    interpolate = LinearNDInterpolator(target_xy, source_xy - target_xy, fill_value=0.)
    return np.asarray(interpolate(grid), dtype=np.float32)


def warp(tf, x, offsets, name):
    # CPU GatherNd/ScatterNd implements the same bilinear warp with deterministic
    # native gradients; border replication is explicit at the sampling boundary.
    with tf.device('/cpu:0'), tf.name_scope(name):
        batch, height, width, _ = x.shape.as_list()
        xx, yy = tf.meshgrid(tf.range(width, dtype=tf.float32), tf.range(height, dtype=tf.float32))
        sx = tf.clip_by_value(xx[None] + offsets[..., 0], 0., float(width-1))
        sy = tf.clip_by_value(yy[None] + offsets[..., 1], 0., float(height-1))
        x0, y0 = tf.floor(sx), tf.floor(sy)
        x1, y1 = tf.minimum(x0+1, width-1.), tf.minimum(y0+1, height-1.)
        bi = tf.tile(tf.reshape(tf.range(batch), [batch, 1, 1]), [1, height, width])
        def sample(px, py):
            return tf.gather_nd(x, tf.stack([bi, tf.cast(py, tf.int32), tf.cast(px, tf.int32)], -1))
        dx, dy = (sx-x0)[..., None], (sy-y0)[..., None]
        return ((1-dx)*(1-dy)*sample(x0,y0) + dx*(1-dy)*sample(x1,y0) +
                (1-dx)*dy*sample(x0,y1) + dx*dy*sample(x1,y1))


def synthesis(tf, live, source, additive, mask):
    return (1-mask) * (live+additive) + mask * source
