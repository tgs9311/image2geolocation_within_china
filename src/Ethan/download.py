from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', cache_dir='/root/autodl-tmp/Ethan/Qwen-VL/Qwen-VL-Models')