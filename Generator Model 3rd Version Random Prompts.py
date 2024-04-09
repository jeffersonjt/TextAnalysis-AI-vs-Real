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

model.load_state_dict(torch.load("model_state_random_train.pt",map_location=torch.device('cpu')))

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

df = pd.read_csv("amazon_review_train_100k.csv",header=None)
reviews = df[0].tolist()
reviews = [str(review) for review in reviews]


def get_words(text):
    words = text.split(" ")
    rng = [1,2,3]
    
    output = words[0]
    
    for i in range(random.choice(rng)):
        output += " "
        output += words[i+1]
        
    return output


random_text = random.choice(reviews)
print("Original Text :\n",random_text,sep = "")
print()
prompt = get_words(random_text)
print("Prompt for Generation:\n",prompt,sep = "")
out = infer(prompt)
print()
print("Generated Text:\n",out.replace("<|endoftext|>",""),sep = "")
                             
