import re



def get_codes_dict():
	file = open('irma_code.txt', 'r')
	codes = {}
	for x in file.readlines():
		if x=='\n':
			continue
		words = x.strip().split(' ')
		# print words
		for word in words:
			if re.match('\d', word[0]):
				# print "here"
				codes[word] = ' '.join(words[1:])
				pw = word
				break
			else:
				# print "here else"
				codes[pw] = codes[pw]+' '.join(words[0:])
				break
			
	return codes

def get_words_from_code(irma, codes):
	words = []
	for i in range(0,len(irma)+1):
		if str(irma)[0:i] in codes:
			words.append(codes[(irma)[0:i]])
	print words


def get_vocab(codes):

	vocab = {}
	for key,words in codes.iteritems():
		# word_list = words.replace(",", " ")
		word_list = re.sub('[^0-9a-zA-Z]+', ' ', words).strip()
		# print "New Words "+word_list
		word_list = word_list.split(' ')
		for w in word_list:
			if w not in vocab:
				if w == ' ':
					continue
				# print "Appending "+w
				vocab[w] = len(vocab)

	return vocab



codes = get_codes_dict()
print codes
# irma = str(raw_input("Enter the IRMA Code\n"))
# get_words_from_code(irma, codes)
vocab = get_vocab(codes)
print (vocab)