"""Update energy in source-weight singular subspaces, including rotations/nullspace."""
import json
import torch


def analyze(source, final, root):
    records=[]
    for name,w in source.items():
        if not name.startswith('backbone.encoder.blocks.') or w.ndim!=2:
            continue
        a=w.double();delta=final[name].double()-a
        u,s,vh=torch.linalg.svd(a,full_matrices=False)
        left=u.T@delta;right=delta@vh.T;core=left@vh.T
        energy=delta.square().sum().item();base=a.square().sum().item()
        assert energy>0
        # Four equal-rank bins, ordered largest -> smallest SOURCE singular values.
        ranks=torch.arange(len(s));bins=torch.tensor_split(ranks,4)
        le=[left[ix].square().sum().item() for ix in bins]
        re=[right[:,ix].square().sum().item() for ix in bins]
        grid=[[core[ix][:,iy].square().sum().item()/energy for iy in bins] for ix in bins]
        left_null=max(0.,energy-sum(le));right_null=max(0.,energy-sum(re))
        assert abs(sum(le)+left_null-energy)<1e-9*max(1.,energy)
        assert abs(sum(re)+right_null-energy)<1e-9*max(1.,energy)
        record=dict(name=name,shape=list(w.shape),rank=len(s),update_energy=energy,source_energy=base,
            relative_update_norm=(energy/base)**.5,left_quartiles=[x/energy for x in le],
            right_quartiles=[x/energy for x in re],left_null_fraction=left_null/energy,
            right_null_fraction=right_null/energy,core_grid=grid,
            singular_min=s[-1].item(),singular_max=s[0].item(),
            diagonal_core_fraction=core.diagonal().square().sum().item()/energy)
        records.append(record)
        (root/'SVD_STATUS.json').write_text(json.dumps(dict(completed=len(records),last=name)))
    total=sum(x['update_energy'] for x in records)
    def weighted(field):
        return sum(x[field]*x['update_energy'] for x in records)/total
    left=[sum(x['left_quartiles'][i]*x['update_energy'] for x in records)/total for i in range(4)]
    right=[sum(x['right_quartiles'][i]*x['update_energy'] for x in records)/total for i in range(4)]
    result=dict(matrices=len(records),update_energy=total,
        relative_update_norm=(total/sum(x['source_energy'] for x in records))**.5,
        left_quartiles=left,right_quartiles=right,left_null_fraction=weighted('left_null_fraction'),
        right_null_fraction=weighted('right_null_fraction'),diagonal_core_fraction=weighted('diagonal_core_fraction'),
        note='Bins by source singular-value rank, equal dimension. Energy fractions, not accuracy attribution. Thin-SVD null complements explicitly retained.')
    (root/'SVD_LAYERS.json').write_text(json.dumps(records,indent=2))
    (root/'SVD_SUMMARY.json').write_text(json.dumps(result,indent=2))
    return result
