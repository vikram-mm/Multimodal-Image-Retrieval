import re
import pandas as pd
import numpy as np
from irma_reader import *


def get_textual_features():

	img = pd.read_csv("ImageCLEFmed2009_train_codes.02.csv")
	
	codes = get_codes_dict('irma_code.txt')
	codes_c = get_codes_dict('IRMA_C.txt')
	vocab = get_vocab(codes,codes_c)

	print "Vocab Size : ",len(vocab),"Number of Images : ",len(img["irma_code"])

	vocab_size = len(vocab)
	num_images = len(img["irma_code"])

	# textual_words = np.zeros((num_images,vocab_size))
	textual_words = {}

	# i = 0
	# total = 0

	path = "dataset/ImageCLEFmed2009_train.02/"

	for img_code,img in zip(img["image_id"],img["irma_code"]):
		
		a,b,c,d = img.split("-")
		# print c,d

		words = set()

		while len(d):
			word = codes.get(d,"unk")
			if word != "unk" and word != "unspecified":
				words.update(word.split(" "))
			d = d[:-1]
			# print d

		while len(c):
			word = codes_c.get(c,"unk")
			if word != "unk" and word != "unspecified":
				words.update(word.split(" "))
			c = c[:-1]

		# print words
		# print path+str(img_code)
		# textual_words[path+str(img_code)] = np.zeros(vocab_size)

		
		textual_words[path+str(img_code)] = [vocab[w] for w in words]

		# print textual_words[path+str(img_code)]
		# break
		# i += 1

	print "Dict len: ",len(textual_words)

	# print total
	# print np.sum(textual_words)
	return textual_words,vocab_size


if __name__=='__main__':

	get_textual_features()
			