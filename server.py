import logging
import random
import torch
from flask import Flask
from flask import render_template
from transformers import GPT2Tokenizer, GPT2LMHeadModel


app = Flask(__name__)

device =  "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained('gpt2') # Also in -large
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to(device)


def get_next_word_probs(prefix):
	""" Given a prefix, returns the probabilities for the following
	possible tokens.

	Parameters
	----------
	prefix : str
		Prefix to use for the text generation

	Returns
	-------
	torch.tensor
		Torch with the probabilities for all possible next tokens.
	"""
	# https://rycolab.io/classes/acl-2023-tutorial/
	input_ids = prefix.to(device)
	with torch.no_grad():
		logits = model(input_ids).logits.squeeze()[-1]
	probabilities = torch.nn.functional.softmax(logits, dim=0)
	return probabilities

@app.route("/gpt2_probs/<sentence>")
def gpt2_probs(sentence):
	input_ids = tokenizer.encode(sentence, return_tensors='pt')
	retval = []
	prefix = []
	for idx, token in enumerate(input_ids[0]):
		word = tokenizer.decode(token).replace(' ', '_')
		if idx<2:
			# The first two words are fully independent
			retval.append({'word': word, 'prob': 1.0,
			               'next_best': [{'word': '-', 'prob': 1.0} for _ in range(5)]})

		else:
			# From the third word on it gets better
			probs = get_next_word_probs(input_ids[0][:idx])

			# Obtain the probabilities of the current word 
			next_word_as_token = input_ids[0][idx]
			next_word_as_word = tokenizer.decode(next_word_as_token)
			next_prob = probs[next_word_as_token].item()
			# Obtain the probabilities for the next words
			next_best = []
			top_token_probs, top_token_vals = torch.topk(probs, 5)
			for token, prob in zip(top_token_vals, top_token_probs):
				next_best.append({'word': tokenizer.decode(token).replace(' ', '_'),
				                  'prob': prob.item()})
			retval.append({'word': next_word_as_word.replace(' ', '_'),
							'prob': next_prob,
							'next_best': next_best})
	"""
	for token in input_ids[0][1:]:
		prefix.append(token)
		str_prefix = ''.join(map(lambda x: tokenizer.decode(x), prefix))
		word = tokenizer.decode(token)

		prob = random.random()
		next_best = []
		for _ in range(5):
			rand_word = tokenizer.decode([random.randint(0,5000)]).replace(' ', '_')
			rand_prob = random.random()
			next_best.append({'word': rand_word, 'prob': rand_prob})
		retval.append({'word': word, 'prob': prob, 'next_best': next_best})
	"""
	return retval

@app.route("/")
def main_screen():
    return render_template('index.html')
