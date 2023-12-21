import logging
import random
import torch
from flask import Flask, render_template, request
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelWithLMHead, set_seed

set_seed(16)
app = Flask(__name__)

device =  "cuda:0" if torch.cuda.is_available() else "cpu"
# English
#tokenizer = GPT2Tokenizer.from_pretrained('gpt2') # Also in -large
#model = GPT2LMHeadModel.from_pretrained('gpt2')
# German
tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
model = AutoModelWithLMHead.from_pretrained("dbmdz/german-gpt2")
model.to(device)


def get_next_word_probs(prefix):
	""" Given a prefix, returns the probabilities for the following
	possible tokens.

	Parameters
	----------
	prefix : torch.tensor
		Tensor containing the tokens used as prefix to generate the
		next words.

	Returns
	-------
	torch.tensor
		Tensor with the probabilities for all possible next tokens.
	"""
	# https://rycolab.io/classes/acl-2023-tutorial/
	input_ids = prefix.to(device)
	with torch.no_grad():
		logits = model(input_ids).logits.squeeze()[-1]
	probabilities = torch.nn.functional.softmax(logits, dim=0)
	return probabilities


@app.route("/gpt2_probs",  methods=['GET'])
def gpt2_probs():
	""" Given a sentence, returns a list of the probabilities for each
	individual word plus a list of the words that were more likely to
	be chosen at any given time.

	Parameters
	----------
	sentence : str
		Sentence that will be used for prediction.

	Returns
	-------
	list(dict())
		A list of dictionaries where every dictionary corresponds to a
		token in the original sentence. Every dictionary contains the
		following keys:
		  * 'word': The word that the token represents
		  * 'prob': The probability of this token given the prefix.
		  * 'next_best': An ordered list of the next best tokens for the
		                 given prefix.

	Notes
	-----
	All spaces have been replaced with '_' to make display easier to
	understand.
	"""
	sentence = request.args.get('sentence', default='No sentence given', type=str)
	input_ids = tokenizer.encode(sentence, return_tensors='pt')
	retval = []
	prefix = []
	for idx, token in enumerate(input_ids[0]):
		word = tokenizer.decode(token).replace(' ', '_')
		if idx<2:
			# The first two words are fully independent, and therefore
			# get no probability.
			retval.append({'word': word, 'prob': 1.0,
			               'next_best': [{'word': '-', 'prob': 1.0} for _ in range(5)]})

		else:
			# From the third word on it gets easier.
			# Obtain the probabilities of the current word 
			probs = get_next_word_probs(input_ids[0][:idx])
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
	return retval


@app.route("/")
def main_screen():
    return render_template('index.html')
