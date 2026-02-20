import torch
from transformers import AutoTokenizer, AutoModel

path = "/home/jovyan/nfs_share/models/InternVL3_5-GPT-OSS-20B-A4B-Preview"
model = AutoModel.from_pretrained(
    path,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False, fix_mistral_regex=True)