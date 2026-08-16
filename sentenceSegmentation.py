from util import *

# Add your import statements here
import re
import nltk
import spacy
from nltk.tokenize import sent_tokenize


class SentenceSegmentation():

	def __init__(self):
		# Load spaCy model (students may use this if needed)
		self.nlp = spacy.load("en_core_web_sm")

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""


		segmentedText=[]	# list to store segmented sentences

		res=""				# temporary string to hold current sentence

		for i in text:		# for every character in the text
			
			res=res+i		# adding that character to current sentence
			
			if i in ['.','?','!']:	# checking if the character is a ending punctuation
			
				segmentedText.append(res.strip())	# if yes add the current sentence to segmented sentence after removing extra spaces
			
				res=""		# make the temporary string empty
		
		if res.strip()!="":		# check if any text remains that does not end with punctuation 
		
			segmentedText.append(res.strip())	# add that text to segmented sentence
		
		return segmentedText	# return segmented sentences


	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""



		segmentedText=[]	# list to store segmented sentences

		segmentedText=sent_tokenize(text)	# use NLTK's pretrained function to split sentence

		return segmentedText	# return segmented sentences


	def spacySegmenter(self, text):
		"""
		Sentence Segmentation using spaCy

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText=[]	# list to store segmented sentences

		doc=self.nlp(text)	# processing the text using spaCy NLP pipeline

		for x in doc.sents:	# for every detected sentences
		
			segmentedText.append(x.text.strip())	# add the sentence to segmented sentence after removing extra spaces

		return segmentedText	# return segmented sentences