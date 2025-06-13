from transformers import AutoModel, AutoProcessor
model = AutoModel.from_pretrained("openai/clip-vit-large-patch14-336")
model.save_pretrained("./models/clip-vit-large-patch14-336")

processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
processor.save_pretrained("./models/clip-vit-large-patch14-336")