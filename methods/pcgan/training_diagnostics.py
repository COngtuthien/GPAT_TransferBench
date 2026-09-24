"""Read-only numerical evidence helpers, separated from optimizer execution."""
import hashlib

from .training_runner import require
from .training_losses import G_TERMS, D_TERMS


def tensor_record(t):
    import numpy as np
    a = t.detach().cpu().contiguous().numpy()
    finite = bool(np.isfinite(a).all())
    require(finite, 'finite diagnostic tensor')
    return dict(shape=list(a.shape), dtype=str(a.dtype), finite=finite,
                sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                nonzero=int(np.count_nonzero(a)), min=float(a.min()), max=float(a.max()),
                mean=float(a.mean(dtype='float64')),
                l2=float(np.linalg.norm(a.astype('float64').ravel())))


def parameters(models):
    return {component: {name: tensor_record(p) for name, p in model.named_parameters()}
            for component, model in models.items()}


def gradient_inventory(named, grads):
    entries = {name: None if g is None else tensor_record(g) for (name, _), g in zip(named, grads)}
    return dict(total=len(entries), non_none=sum(v is not None for v in entries.values()),
                finite=sum(v is not None and v['finite'] for v in entries.values()),
                nonzero=sum(v is not None and v['nonzero'] > 0 for v in entries.values()),
                entries=entries)


def connectivity(loss, models, inputs, *, retain_graph):
    import torch
    groups = {key: list(model.named_parameters()) for key, model in models.items()}
    groups.update({key: [(key, value)] for key, value in inputs.items()})
    named = [(group + '/' + n, p) for group, items in groups.items() for n, p in items]
    grads = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True,
                                retain_graph=retain_graph)
    result, offset = {}, 0
    for group, items in groups.items():
        result[group] = gradient_inventory(items, grads[offset:offset+len(items)])
        offset += len(items)
    return result


def validate_connectivity(records, *, generator):
    for term, groups in records.items():
        if generator:
            connected = {'Encoder', 'Generator', 'x_src', 'z_pat_src'}
            if term in ('L_rec', 'L_advrec'):
                connected.add('z_con_src')
            else:
                connected.update(('x_tgt', 'z_con_tgt'))
            if term in ('L_advrec', 'L_advmix'):
                connected.add('ImageD')
            if term == 'L_pat':
                connected.add('PatchD')
        else:
            connected = {'ImageD' if term in D_TERMS[:3] else 'PatchD'}
            if term == 'L_D_real':
                connected.update(('x_src', 'x_tgt'))
            if term in D_TERMS[3:]:
                connected.add('x_src')
        for name, inventory in groups.items():
            require(inventory['finite'] == inventory['non_none'], 'finite connectivity')
            if name in connected:
                require(inventory['nonzero'] > 0, term + ' connected to ' + name)
            else:
                require(inventory['non_none'] == 0, term + ' disconnected from ' + name)


def external_totals(terms):
    """Materialize individual FP32 scalars; independently sum outside the graph."""
    import numpy as np
    f = {name: np.float32(value.detach().cpu().item()) for name, value in terms.items()}
    require(all(value.dtype.is_floating_point for value in terms.values()), 'floating losses')
    result = {}
    if 'L_G_total' in f:
        total = f[G_TERMS[0]]
        for name in G_TERMS[1:]:
            total = np.float32(total + f[name])
        result['L_G_total'] = total
    else:
        image = np.float32(np.float32(f['L_D_real'] + np.float32(.5) * f['L_D_rec'])
                           + np.float32(.5) * f['L_D_mix'])
        patch = np.float32(f['L_D_patch_real'] + f['L_D_patch_fake'])
        result.update(L_D_image=image, L_D_patch=patch, L_D_total=np.float32(image + patch))
    evidence = {}
    for name, value in result.items():
        require(value.tobytes() == f[name].tobytes(), 'exact external FP32 equality ' + name)
        evidence[name] = dict(value=float(value), fp32_hex=value.tobytes().hex(),
                              runner_fp32_hex=f[name].tobytes().hex(), exact=True)
    return dict(scalars={k: float(v) for k, v in f.items()}, external=evidence)


def step_gradients(groups):
    return {key: gradient_inventory(named, [p.grad for _, p in named]) for key, named in groups.items()}


def changes(before, after, gradients, active):
    counts, detail = {}, {}
    for component, entries in before.items():
        changed = zero_unchanged = nonzero_unchanged = 0
        detail[component] = {}
        for name, initial in entries.items():
            final = after[component][name]
            is_changed = initial['sha256'] != final['sha256']
            gradient = gradients[active]['entries'].get(component + '.' + name)
            nonzero = gradient is not None and gradient['nonzero'] > 0
            changed += is_changed
            zero_unchanged += not is_changed and not nonzero
            nonzero_unchanged += not is_changed and nonzero
            detail[component][name] = dict(changed=is_changed, nonzero_gradient=nonzero)
        counts[component] = dict(total=len(entries), changed=changed,
                                 unchanged_with_zero_gradient=zero_unchanged,
                                 unchanged_with_nonzero_gradient=nonzero_unchanged)
        owns = component in (('Encoder', 'Generator') if active == 'G' else ('ImageD', 'PatchD'))
        require(changed > 0 if owns else changed == 0, 'step isolation/change ' + component)
    return dict(counts=counts, parameters=detail,
                rounding_note='A nonzero gradient can produce an FP32-rounded unchanged parameter; every identity is retained.')


def adam_state(optimizer, named):
    import torch
    owners = {id(p): name for name, p in named}
    require(all(id(p) in owners for p in optimizer.state), 'Adam state ownership')
    entries = {}
    for p, state in optimizer.state.items():
        require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Adam state structure')
        require(state['step'].item() == 1, 'exactly one Adam application')
        require(state['exp_avg'].shape == state['exp_avg_sq'].shape == p.shape, 'Adam state shape')
        require(state['exp_avg'].dtype == state['exp_avg_sq'].dtype == torch.float32, 'FP32 Adam')
        entries[owners[id(p)]] = {k: tensor_record(v) for k, v in state.items()}
    require(set(entries) == set(owners.values()), 'every expected Adam state')
    return dict(count=len(entries), entries=entries, ownership_verified=True)
