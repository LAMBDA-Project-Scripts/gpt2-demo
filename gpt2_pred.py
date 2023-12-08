from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

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
	input_ids = tokenizer.encode(prefix, return_tensors='pt').to(device)

	with torch.no_grad():
		logits = model(input_ids).logits.squeeze()[-1]
	probabilities = torch.nn.functional.softmax(logits, dim=0)
	return probabilities


if __name__ == '__main__':
	"""
	text = "It flies like a crow."
	print(tokenizer.encode('crow'))
	print(tokenizer.encode(' crow'))
	"""
	text = "I spread the butter on my bread with my knife"
	words = text.split()
	for idx, word in enumerate(words):
		if idx==0:
			continue
		prefix = " ".join(words[:1+idx])
		print(prefix)

		# Calculate probabilities for the next word
		probs = get_next_word_probs(prefix)
		top_token_probs, top_token_vals = torch.topk(probs, 20)
		for token, prob in zip(top_token_vals, top_token_probs):
			print(f"  {prob.item():.3f} '{tokenizer.decode(token)}'")

		# Calculate probabilities for *our* next word
		try:
			next_word = f" {words[idx+1]}"
			next_word_as_token = tokenizer.encode(next_word)
			if len(next_word_as_token) > 1:
				print(f"Word '{next_word}' cannot be encoded with a single token")
				as_tokens = list(map(lambda x: tokenizer.decode(x), next_word_as_token))
				print(f"Encoding: {as_tokens}")
				next_word_as_token = next_word_as_token[0]
		except IndexError:
			next_word = "<eos>"
			next_word_as_token = 50256
		token_prob = probs[next_word_as_token].item()
		print(f"  *{token_prob:.6f} '{next_word}'")

