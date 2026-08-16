from util import *

# Add your import statements here

import nltk
from nltk.corpus import stopwords


class StopwordRemoval():

	def __init__(self):
		self.stop_words = set(stopwords.words('english'))

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stopwordRemovedText=[]	#  list to store stopwords

		for i in text:	# for each sentence in the text

			sw = []	# list to store stopwords for current sentence

			for j in i:	# for each word in the sentence

				if j.lower() not in self.stop_words:	# convert word  to lowercase and checks whether the word is a stopword or not

					sw.append(j)	# add stopwords of current sentence to list

			stopwordRemovedText.append(sw)	# add stopwords to main list

		return stopwordRemovedText	# return stopwords list




	