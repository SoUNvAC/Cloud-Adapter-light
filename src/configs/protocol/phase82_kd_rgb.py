"""Exactly Phase80 student/data/optimizer, plus fixed conditional KD in loss."""
_base_=['./phase80_lora_rgb.py']
model=dict(type='Phase82ConditionalKDStudent')
work_dir='./work_dirs/phase82_20261011/lora_seed65'
