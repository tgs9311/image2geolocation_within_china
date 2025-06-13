### Introduction of Ethan
```
cd Ethan
git clone https://github.com/QwenLM/Qwen-VL.git
cd Qwen-VL
pip install -r requirements.txt
mkdir Qwen-VL-Models 
mkdir LoRA
```
- Then download the pre-trained LVLM weights into the `Qwen-VL-Models` folder and the LoRA weights into the `LoRA` folder.
  ```Python
  python infer.py # with the test image
  # Due to the inherent randomness in LVLM generation, the generated reasons may not always be consistent.
  ```
- Training steps (Reasoning Tuning Phase)
```
cd Ethan
git clone https://github.com/QwenLM/Qwen-VL.git
cd Qwen-VL
pip install -r requirements.txt
mkdir Qwen-VL-Models 
mkdir LoRA
mkdir Dataset
```
- Then download the pre-trained LVLM weights into the `Qwen-VL-Models` folder and the SFT data into the `Dataset` folder.
```
mv finetune_lora_reason.sh Qwen-VL/finetune
cd Qwen-VL
sh finetune/finetune_lora_reason.sh
```