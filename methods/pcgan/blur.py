"""Explicit future A5 pooling hook. No Torch import during static preparation."""
from methods.common.learned import PreparationError
from .contract import validate_contract


def blur_mapping(config=None):
    return validate_contract(config)['overlay']['blur_operator']


def blur_inputs(target, generated, *, config=None):
    """Apply A5 independently without detaching either branch or any fallback.

    Tensor/module availability is checked only for this explicit execution hook.
    Static plans never call it. No model is created, loaded, or trained here.
    """
    op = blur_mapping(config)
    import torch
    expected = (op['input_resolution'],) * 2
    for tensor in (target, generated):
        if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 4 or
                tuple(tensor.shape[-2:]) != expected or not tensor.is_floating_point()):
            raise PreparationError('A5 requires floating NCHW tensors at canonical resolution')
    if target.shape != generated.shape:
        raise PreparationError('target and generated shapes must match')
    kwargs = {key: op[key] for key in
              ('kernel_size', 'stride', 'padding', 'ceil_mode', 'count_include_pad')}
    return (torch.nn.functional.avg_pool2d(target, **kwargs),
            torch.nn.functional.avg_pool2d(generated, **kwargs))
