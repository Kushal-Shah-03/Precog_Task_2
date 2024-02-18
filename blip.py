import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cpu")

# img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open("data/img/01236.png").convert('RGB')

# conditional image captioning
text = "hate"

inputs = processor(raw_image, text, return_tensors="pt").to("cpu")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

text = "love"


inputs = processor(raw_image, text, return_tensors="pt").to("cpu")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

# unconditional image captioning
inputs = processor(raw_image, return_tensors="pt").to("cpu")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
