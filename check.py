from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import torch

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenized input
text = "What is your name ? Donald Trump is my given name"
tokenized_text = tokenizer.tokenize(text)

# Mask a token that we will try to predict back with BertForMaskedLM
masked_index = 8
tokenized_text[masked_index] = '[MASK]'
print(tokenized_text)
assert tokenized_text == ['what', 'is', 'your', 'name', '?', 'donald', 'trump', 'is', '[MASK]', 'given', 'name']

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

print(indexed_tokens)# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# Predict hidden states features for each layer
encoded_layers, _ = model(tokens_tensor, segments_tensors)
# We have a hidden states for each of the 12 layers in model bert-base-uncased
assert len(encoded_layers) == 12
# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

# Predict all tokens
predictions = model(tokens_tensor, segments_tensors)

# confirm we were able to predict 'henson'
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])

print(predicted_token)