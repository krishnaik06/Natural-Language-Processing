# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 22:19:40 2019

@author: Krish.Naik
"""

import nltk

paragraph =  """I have three visions for India. In 3000 years of our history, people from all over 
               the world have come and invaded us, captured our lands, conquered our minds. 
               From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British,
               the French, the Dutch, all of them came and looted us, took over what was ours. 
               Yet we have not done this to any other nation. We have not conquered anyone. 
               We have not grabbed their land, their culture, 
               their history and tried to enforce our way of life on them. 
               Why? Because we respect the freedom of others.That is why my 
               first vision is that of freedom. I believe that India got its first vision of 
               this in 1857, when we started the War of Independence. It is this freedom that
               we must protect and nurture and build on. If we are not free, no one will respect us.
               My second vision for India’s development. For fifty years we have been a developing nation.
               It is time we see ourselves as a developed nation. We are among the top 5 nations of the world
               in terms of GDP. We have a 10 percent growth rate in most areas. Our poverty levels are falling.
               Our achievements are being globally recognised today. Yet we lack the self-confidence to
               see ourselves as a developed nation, self-reliant and self-assured. Isn’t this incorrect?
               I have a third vision. India must stand up to the world. Because I believe that unless India 
               stands up to the world, no one will respect us. Only strength respects strength. We must be 
               strong not only as a military power but also as an economic power. Both must go hand-in-hand. 
               My good fortune was to have worked with three great minds. Dr. Vikram Sarabhai of the Dept. of 
               space, Professor Satish Dhawan, who succeeded him and Dr. Brahm Prakash, father of nuclear material.
               I was lucky to have worked with all three of them closely and consider this the great opportunity of my life. 
               I see four milestones in my career"""
               
               
# Cleaning the texts (DATA CLEANING)
import re # for regular expression
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer() #we will  generally perform lemmatization only tho
wordnet=WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = [] #reason- After cleaning the text we will keep the text here
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i]) #apart from a-z or A-Z re.sub replaces all characters like ", . / ?" with" "; sentences[i]) means applying for each sentences gathered after tokenization like arr[i]
    review = review.lower() # lowwering the sentence
    review = review.split() #.split means() means splitting up the sentences into list of words basically word tokenizastion
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] # now if the words are not stopwords they are stemmatized and the stopwords are removed
    review = ' '.join(review) #finally words rejoin to give the resultant sentence
    corpus.append(review) # this sentences we put in the blank list "corpus"
    
# Creating the Bag of Words model (FEATURE EXTRACTION)
from sklearn.feature_extraction.text import CountVectorizer # CountVectorizer for creating BOW from scikit learn
cv = CountVectorizer(max_features = 1500)  #cv is just an object for CountVectorizer (max_features = 1500---limiting the max no of features that will be created after BOW to 1500)
X = cv.fit_transform(corpus).toarray() #cv.fit_transform(list of sentences after sent_tokenize of paragraph).toarray()






