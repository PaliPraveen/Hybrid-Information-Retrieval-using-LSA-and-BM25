# # Add your import statements here






# # Add any utility functions here



###########       Question 4.3        ###############


# import json
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from collections import Counter

# with open(r"C:\Users\palip\Desktop\cran_docs.json") as f:   # loading Cranfield documents
#     docs = json.load(f)

# documents = [doc["body"] for doc in docs]     # extracting document text 

# #performing tokenization
# final_token = []
# for doc in documents:
#     tokens = word_tokenize(doc.lower())
#     for token in tokens:
#         if token.isalpha():
#             final_token.append(token)

# # counting word frequencies
# word_counts = Counter(final_token)
# total_words = sum(word_counts.values())

# # using threshold value as 0.002
# threshold = 0.002

# #performing bottom up approach
# sw = set()
# for word, count in word_counts.items():
#     if count / total_words > threshold:
#         sw.add(word)

# # stopwords in NLTK
# nltk_sw = set(stopwords.words("english"))

# # comparison between stopwords in bottom up approach and in NLTK
# overlap = sw.intersection(nltk_sw)
# only_bottomup = sw - nltk_sw
# only_nltk = nltk_sw - sw

# print("Number of stopwords in bottom up approach:", len(sw))
# print("Number of NLTK stopwords:", len(nltk_sw))
# print("Overlap between lists:", len(overlap))

# print("\nstopwords in bottom up approach:")
# print(sorted(sw))

# print("\nstopwords in NLTK:")
# print(sorted(nltk_sw))

# print("\nWords only in bottom up approach:")
# print(sorted(only_bottomup))

# print("\nWords only in NLTK stopwords:")
# print(sorted(only_nltk))