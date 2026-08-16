from util import *

# Add your import statements here
# (Students may import required libraries such as nltk, spacy, re, etc.)

import nltk
import spacy
from nltk.tokenize import TreebankWordTokenizer
import re


class Tokenization():


	def __init__(self):

		self.nlp = spacy.load("en_core_web_sm")

		self.tokenizer = TreebankWordTokenizer()

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText=[]	# list to store tokens

		for i in text:	# for each sentence in the text

			words = i.split()	# split sentence by space to get words

			tokens = []		# list to store tokens for current sentence

			for j in words:	# process each word
			
				token = re.findall(r"[A-Za-z0-9]+|[.,\-!?;:]", j) 	# using regex to seperate words and punctuations
			
				tokens.extend(token)	# adding the extracted tokens to the token list

			tokenizedText.append(tokens) # adding tokens to the tokenizedText

		return tokenizedText	# return tokens



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText=[]	# list to store tokens

		for i in text:	# for each sentence in the text

			token = self.tokenizer.tokenize(i) # generate tokens using penn treebank tokenizer function

			tokenizedText.append(token)	# add tokens to the tokenizedText list

		return tokenizedText	# return tokens



	def spacyTokenizer(self, text):
		"""
		Tokenization using spaCy

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText=[]	# list to store tokens

		for i in text:	# for each sentence in the text

			doc = self.nlp(i)	# applying spaCy NLP pipeline
	
			tokens = [token.text for token in doc]	# extract tokens from document

			tokenizedText.append(tokens)	# add tokens to the list

		return tokenizedText	# return tokens
