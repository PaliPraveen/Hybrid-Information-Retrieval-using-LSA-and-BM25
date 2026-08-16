from util import *

# Add your import statements here
# (Students may import required libraries such as nltk, WordNetLemmatizer, PorterStemmer, etc.)
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

class InflectionReduction:


	def __init__(self):
		self.stemmer = PorterStemmer()
		self.lemmatizer = WordNetLemmatizer()

	def porterStemmer(self, text):
		"""
		Inflection Reduction using Porter Stemmer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed tokens representing a sentence
		"""

		reducedText=[]	# list to store stemmed sentences

		for i in text:	# for each sentence in the text

			ss = []	# list to store stemmed words of current sentence

			for j in i:	# for each word in the sentence

				stem = self.stemmer.stem(j)	# apply porter stemming to the word

				ss.append(stem)	# add the stemmed word to the list

			reducedText.append(ss)	# add the sub list to main list

		return reducedText	# return reducedtext



	def wordnetLemmatizer(self, text):
		"""
		Inflection Reduction using WordNet Lemmatizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			lemmatized tokens representing a sentence
		"""

		reducedText=[]	# list to store lematized sentences

		for i in text:	# for each sentence in the text

			ls = []	# # list to store lematized words of current sentence

			for j in i:	# for each word in the sentence

				lemma = self.lemmatizer.lemmatize(j)	# apply wordnet lemmatizer to the word

				ls.append(lemma)	# add the lemmatized word to the list

			reducedText.append(ls)	# add the sub list to main list

		return reducedText	# return reducedtext



	def reduce(self, text):
		"""
		Wrapper function for inflection reduction.
		Students may choose which method to call
		or extend this function to support both options.
		"""
		reducedText=[]

		reducedText = self.wordnetLemmatizer(text) # call the lemmatizer method

		return reducedText
