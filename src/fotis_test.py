from transformers import GPT2Tokenizer, TFGPT2Model
import tensorflow as tf

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = TFGPT2Model.from_pretrained('gpt2', return_dict=True)
inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
outputs = model(inputs)
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states)
