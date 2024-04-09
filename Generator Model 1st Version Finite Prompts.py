import random
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging

logger = logging.getLogger("transformers")
logger.setLevel(logging.ERROR)



device = "cpu" 
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2")
model = model.to(device)

model.load_state_dict(torch.load("model_state.pt",map_location=torch.device('cpu')))

def infer(inp):
    inp = tokenizer(inp,return_tensors = "pt");
    X = inp["input_ids"].to(device) 
    a = inp["attention_mask"].to(device)
    output = model.generate(X,attention_mask = a,
                            max_length=300,
                            do_sample=True,
                            top_p=0.95,
                            top_k=50,
                            temperature=1.0,
                            num_return_sequences=1)
    output = tokenizer.decode(output[0])
    return output



prompts = ["I bought this product and",
"I have been using this product for",
"I was impressed with the",
"I was disappointed with the",
"The best thing about this product is",
"The worst thing about this product is",
"If you're looking for",
"I would recommend this product to",
"I wouldn't recommend this product to",
"I was pleasantly surprised by",
"I was not impressed with",
"This product exceeded my expectations",
"This product did not meet my expectations",
"The price is",
"The quality is",
"The packaging was",
"The shipping was",
"I've never used a product like this before",
"I've used similar products before, but this one is"]


inp = random.choice(prompts)
out = infer(inp)
out = out.replace("<|endoftext|>","")
print("Prompt:")
print(inp)
print()
print("Generated Text:")      
print(out)                   
