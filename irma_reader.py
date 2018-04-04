import re



def get_codes_dict(path):
	file = open(path, 'r')
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


def get_vocab(codes,codes2=None):

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

	if(codes2 != None):

		for key,words in codes2.iteritems():
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


if __name__=='__main__':

	codes_d = get_codes_dict('irma_code.txt')
	# print codeseses
	for key in sorted(codes_d):
   		print key,' ',codes_d[key]
	# irma = str(raw_input("Enter the IRMA Code\n"))
	# get_words_from_code(irma, codes)
	print '***************'
	codes_c = get_codes_dict('IRMA_C.txt')
	for key in sorted(codes_c):
   		print key,' ',codes_c[key]

	vocab = get_vocab(codes_d,codes_c)
	for key in sorted(vocab):
   		print key,' ',vocab[key]
	print (len(vocab))