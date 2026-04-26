CUDA_VISIBLE_DEVICES=0 python -c "
from vllm import LLM
llm = LLM(
    model='/home/jovyan/nfs_share/models/InternVL3_5-8B',
    tensor_parallel_size=1,
    attention_backend='FLASHINFER',
    max_model_len=8192,
    trust_remote_code=True,
    enforce_eager=True,
)
print('Engine initialized OK')
del llm
"