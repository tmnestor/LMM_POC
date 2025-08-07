
import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

model_id = "/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision-Instruct"
# here, specify the name of the image
imageName = "/home/jovyan/nfs_share/tod/datasets/synthetic_invoice_014.png"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

# open the image
image = Image.open(imageName)

messageDataStructure = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": "How much did Jessica pay?",
            },
        ],
    }
]

# create text input
textInput = processor.apply_chat_template(
    messageDataStructure, add_generation_prompt=True
)
# call the processor
inputs = processor(image, textInput, return_tensors="pt").to(model.device)

# here, change the number of tokens to get a more detailed answer
output = model.generate(**inputs, max_new_tokens=2000)
# here, we decode and store the response so we can print it
generatedOutput = processor.decode(output[0])

print(generatedOutput)
