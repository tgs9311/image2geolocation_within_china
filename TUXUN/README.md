# TUXUN

TuXun is a China-specific geolocation system that predicts the geographic location (country, province, city) of an image. It supports fine-tuning on Qwen-VL using Supervised Fine-Tuning (SFT) techniques.

## Train Models

To train the model on different geographic levels:

```bash
# Train country-level model
python train_country.py

# Train province-level model
python train_province.py

# Train city-level model
python train_city.py
```

To fine-tune using instruction-based SFT:
1.Prepare instruction-format data
2.Use the instructions in `Ethan/`.
3.To test the finetune results,use `infer.py`.


## Run the main script(evaluate the model)
```bash
python main.py
```
Run main.py to evaluate the final training results.
## Web Demo
You can launch a local web application to upload images, get predictions, and visualize results on a map.
```bash
python app.py
```
By default, it will start at: http://localhost:5000