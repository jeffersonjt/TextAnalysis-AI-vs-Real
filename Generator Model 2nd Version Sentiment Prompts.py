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

model.load_state_dict(torch.load("model_state_sentiment.pt",map_location=torch.device('cpu')))

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



print(infer("Bad:").replace("<|endoftext|>",""))
print()
print(infer("Good:").replace("<|endoftext|>",""))

                             
