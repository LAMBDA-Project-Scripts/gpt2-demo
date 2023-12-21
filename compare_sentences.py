import numpy as np
import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelWithLMHead, set_seed


set_seed(16)
device =  "cuda:0" if torch.cuda.is_available() else "cpu"
# gpt2, gpt2-large
#tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
model = AutoModelWithLMHead.from_pretrained("dbmdz/german-gpt2")
model.to(device)


def get_text_probs(text):
	""" Given a text, returns the probabilities for each of its
	individual tokens.

	Parameters
	----------
	sentence : torch.tensor
		Tensor containing the tokens for the given input text.

	Returns
	-------
	list(dict())
		A list of all tokens, their conversion to word, and their
		model generation probability.

	Notes
	-----
	Code inspired by https://rycolab.io/classes/acl-2023-tutorial/
	"""
	retval = []
	for i in range(len(text[0])):
		input_ids = text[:, :i].to(device)
		token = text[0][i].item()
		token_dict = {'token': token, 
					  'word': tokenizer.decode(token).replace(' ', '_'),
					  'prob': 1.0 }
		if i > 2:
			with torch.no_grad():
				logits = model(input_ids).logits.squeeze()[-1]
			probabilities = torch.nn.functional.softmax(logits, dim=0)
			token_dict['prob'] = probabilities[token].item()
		retval.append(token_dict)
	return retval


def get_text_perplexity(text):
	""" Given a text, returns the perplexity for each of its
	individual tokens.

	Parameters
	----------
	sentence : torch.tensor
		Tensor containing the tokens for the given input sentence.

	Returns
	-------
	list(dict())
		A list of all tokens, their conversion to word, and their
		model generation probability.

	Notes
	-----
	Code inspired by https://huggingface.co/docs/transformers/perplexity
	"""
	max_length = model.config.n_positions
	# stride = 512
	stride = 16
	#seq_len = sentence.input_ids.size(1)
	seq_len = text.size(1)
	nlls = []
	prev_end_loc = 0
	for begin_loc in range(0, seq_len, stride):
		end_loc = min(begin_loc + max_length, seq_len)
		trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
		#input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
		input_ids = text[:, begin_loc:end_loc].to(device)
		target_ids = input_ids.clone()
		target_ids[:, :-trg_len] = -100

		with torch.no_grad():
			outputs = model(input_ids, labels=target_ids)

			# loss is calculated using CrossEntropyLoss which averages over valid labels
			# N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
			# to the left by 1.
			neg_log_likelihood = outputs.loss

		nlls.append(neg_log_likelihood)

		prev_end_loc = end_loc
		if end_loc == seq_len:
			break
	return torch.exp(torch.stack(nlls).mean()).item()



def needleman_wunsch(x, y, match = 1, mismatch = 1, gap = 1):
	""" Aligns two sequences using the Needleman-Wunsch algorithm.
	
	Parameters
	----------
	x: list(str)
		First sequence to align.
	y: list(str)
		Second sequence to align.

	Returns
	-------
	list(str), list(str)
		The first and second sentences, each containing the required
		markers for alineation.

	Notes
	-----
	This algorithm has been slightly modified. The original
	implementation can be found at
	https://gist.github.com/slowkow/06c6dba9180d013dfd82bec217d22eb5
	"""
	nx = len(x)
	ny = len(y)
	# Optimal score at each possible pair of characters.
	F = np.zeros((nx + 1, ny + 1))
	F[:,0] = np.linspace(0, -nx * gap, nx + 1)
	F[0,:] = np.linspace(0, -ny * gap, ny + 1)
	# Pointers to trace through an optimal aligment.
	P = np.zeros((nx + 1, ny + 1))
	P[:,0] = 3
	P[0,:] = 4
	# Temporary scores.
	t = np.zeros(3)
	for i in range(nx):
		for j in range(ny):
			if x[i] == y[j]:
				t[0] = F[i,j] + match
			else:
				t[0] = F[i,j] - mismatch
			t[1] = F[i,j+1] - gap
			t[2] = F[i+1,j] - gap
			tmax = np.max(t)
			F[i+1,j+1] = tmax
			if t[0] == tmax:
				P[i+1,j+1] += 2
			if t[1] == tmax:
				P[i+1,j+1] += 3
			if t[2] == tmax:
				P[i+1,j+1] += 4
	# Trace through an optimal alignment.
	i = nx
	j = ny
	rx = []
	ry = []
	while i > 0 or j > 0:
		if P[i,j] in [2, 5, 6, 9]:
			rx.append(x[i-1])
			ry.append(y[j-1])
			i -= 1
			j -= 1
		elif P[i,j] in [3, 5, 7, 9]:
			rx.append(x[i-1])
			ry.append('-')
			i -= 1
		elif P[i,j] in [4, 6, 7, 9]:
			rx.append('-')
			ry.append(y[j-1])
			j -= 1
	# Reverse the strings.
	# This is the original code, which I believe is a bug.
	# Also, I want to return a list of individual tokens instead of a
	# single, long string.
	# rx = ''.join(rx[::-1])
	# ry = ''.join(ry[::-1])
	rx = rx[::-1]
	ry = ry[::-1]
	return rx, ry


if __name__ == '__main__':
	source_dir = os.path.join('/', 'home', 'uni', 'Projects', '2023', 'first_protocol', 'stimuli-pro-loud')
	stimuli1 = []
	stimuli2 = []
	with open(os.path.join(source_dir, 'stimulus1.txt'), 'r') as fp:
		for line in fp:
			stimuli1.append(line)
	with open(os.path.join(source_dir, 'stimulus2.txt'), 'r') as fp:
		for line in fp:
			stimuli2.append(line)
	# T
	print(stimuli1[8])
	print(stimuli2[8])
	tokens_l = tokenizer.encode(stimuli1[8], return_tensors='pt')
	tokens_r = tokenizer.encode(stimuli2[8], return_tensors='pt')
	t1 = list(map(lambda x: tokenizer.decode(x), tokens_l[0].numpy()))
	t2 = list(map(lambda x: tokenizer.decode(x), tokens_r[0].numpy()))
	
	aligned_t1, aligned_t2 = needleman_wunsch(t1, t2)
	for at1, at2, in zip(aligned_t1, aligned_t2):
		if at1 != at2:
			print(f'*\t{at1}\t\t{at2}')
		else:
			print(f' \t{at1}\t\t{at2}')

    # Probability and perplexity of each text, side-by-side
	for stim1, stim2 in zip(stimuli1, stimuli2):
		tokens = tokenizer.encode(stim1, return_tensors='pt')
		probs1 = get_text_probs(tokens)
		ppl1 = get_text_perplexity(tokens)
		tokens = tokenizer.encode(stim2, return_tensors='pt')
		probs2 = get_text_probs(tokens)
		ppl2 = get_text_perplexity(tokens)

		mean_t1 = sum(map(lambda x: x['prob'], probs1))/len(probs1)
		mean_t2 = sum(map(lambda x: x['prob'], probs2))/len(probs2)
		# Note: Look at https://stackoverflow.com/questions/63543006/how-can-i-find-the-probability-of-a-sentence-using-gpt-2
		print(f'{mean_t1:.4f}, {mean_t2:.4f} {mean_t1-mean_t2:.4f} {ppl1:.4f} {ppl2:.4f}')
