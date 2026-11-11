"""Strict conversion of the supplied HF S+/16 safetensors to native DINOv3."""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import torch
from dinov3.hub.backbones import dinov3_vits16plus


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('source')
    parser.add_argument('destination')
    args = parser.parse_args()
    raw = Path(args.source).read_bytes()
    length = struct.unpack('<Q', raw[:8])[0]
    header = json.loads(raw[8:8+length])
    hf = {}
    for key, entry in header.items():
        if key == '__metadata__':
            continue
        assert entry['dtype'] == 'F32', (key, entry['dtype'])
        begin, end = entry['data_offsets']
        hf[key] = torch.frombuffer(bytearray(raw[8+length+begin:8+length+end]),
                                   dtype=torch.float32).reshape(entry['shape']).clone()
    model = dinov3_vits16plus(pretrained=False)
    expected = model.state_dict()
    converted, used = {}, set()

    def take(name):
        assert name not in used, name
        used.add(name)
        return hf[name]

    for native, source in {
        'cls_token':'embeddings.cls_token', 'mask_token':'embeddings.mask_token',
        'storage_tokens':'embeddings.register_tokens',
        'patch_embed.proj.weight':'embeddings.patch_embeddings.weight',
        'patch_embed.proj.bias':'embeddings.patch_embeddings.bias',
        'norm.weight':'norm.weight', 'norm.bias':'norm.bias',
    }.items():
        value = take(source)
        converted[native] = value.squeeze(1) if native == 'mask_token' else value
    for i in range(12):
        n, h = f'blocks.{i}.', f'layer.{i}.'
        converted[n+'attn.qkv.weight'] = torch.cat([
            take(h+f'attention.{part}_proj.weight') for part in ('q','k','v')])
        q, v = take(h+'attention.q_proj.bias'), take(h+'attention.v_proj.bias')
        converted[n+'attn.qkv.bias'] = torch.cat([q, torch.zeros_like(q), v])
        for native, source in {
            'attn.proj':'attention.o_proj', 'mlp.w1':'mlp.gate_proj',
            'mlp.w2':'mlp.up_proj', 'mlp.w3':'mlp.down_proj',
            'norm1':'norm1', 'norm2':'norm2',
        }.items():
            for suffix in ('weight','bias'):
                converted[n+native+'.'+suffix] = take(h+source+'.'+suffix)
        for j in (1,2):
            converted[n+f'ls{j}.gamma'] = take(h+f'layer_scale{j}.lambda1')
    assert used == set(hf), sorted(set(hf)-used)
    # Only deterministic architecture buffers may be reconstructed.
    parameter_names = set(dict(model.named_parameters()))
    assert parameter_names <= set(converted), sorted(parameter_names-set(converted))
    generated = []
    for name in set(expected)-set(converted):
        assert name == 'rope_embed.periods' or name.endswith('.attn.qkv.bias_mask'), name
        converted[name] = expected[name].clone()
        generated.append(name)
    assert set(converted) == set(expected)
    for name, value in converted.items():
        assert value.shape == expected[name].shape, (name, value.shape, expected[name].shape)
        assert torch.isfinite(value).all(), name
    model.load_state_dict(converted, strict=True)
    # Verify every supplied tensor survives serialization through the native model.
    for name, value in converted.items():
        assert torch.equal(model.state_dict()[name], value), name
    destination = Path(args.destination)
    assert not destination.exists(), destination
    torch.save(converted, destination)
    reloaded = torch.load(destination, map_location='cpu')
    assert all(torch.equal(reloaded[k], v) for k,v in converted.items())
    receipt = dict(source_sha256=hashlib.sha256(raw).hexdigest(),
        native_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        supplied_tensors=len(hf), native_tensors=len(converted),
        parameters=sum(p.numel() for p in model.parameters()),
        generated_buffers=sorted(generated), key_bias='zero, masked by architecture',
        all_parameters_loaded=True, strict_load=True, serialization_exact=True)
    destination.with_suffix('.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
