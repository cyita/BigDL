set PYTHONPATH=D:\yina\BigDL\python\llm\src
@REM set PYTHONPATH=

set BIGDL_USE_NPU=1

@REM python llama2.py --repo-id-or-model-path "D:\llm-models\Llama-2-7b-chat-hf" --quantization_group_size 0 --n-predict 64 --max-prompt-len 960
python qwen.py --repo-id-or-model-path "D:\llm-models\Qwen2-7B-Instruct" --quantization_group_size 128 --n-predict 64 --max-prompt-len 960