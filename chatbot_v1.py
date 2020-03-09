# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:27:07 2020

@author: c_Rajh
"""
#Importing Libs
import nltk
import random 
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords

## Loadig the file
def loadFile():
    f=open('data/chatbot_db.txt','r',errors = 'ignore')
    raw=f.read()
    raw= raw.lower()
    sent = nltk.sent_tokenize(raw)
    word = nltk.word_tokenize(raw)
    return raw, sent, word

## Creating few default responses
def default_response(ux):
    default_ip = ("hello", "hi", "greetings", "sup", "what's up","hey", 'hey there', 'hey chatbot' )
    default_res = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
    for word in ux.split():
        if word.lower() in default_ip:
            return random.choice(default_res)

## Creating function to tokenize the strings
def tokenize(text):
    lem = nltk.stem.WordNetLemmatizer()
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    stop_words = set(stopwords.words('english'))
    token1 = nltk.word_tokenize(text.lower().translate(remove_punct_dict))
    filtered_sentence = [] 
    for w in token1: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
    token2 = [lem.lemmatize(token) for token in filtered_sentence]
    return token2       
 
## Creating function to find best possible reposnse using tf-idf approach       
def find_response(ux):
    raw, sent, word = loadFile()
    response = ''
    sent.append(ux)
    TfidfVec = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        response =response + "I am sorry! I don't understand you. Could you frame your question is another way?"
        return response
    else:
        response = response + sent[idx]
        return response  

## Main Function
def main():
    flag=True
    print("Bot: Say Hi to me!! If you want to exit, type Bye!\n") 
    while(flag==True):
        user_response = input()
        user_response=user_response.lower()
        if(user_response!='bye'):
            if(default_response(user_response)!=None):
                print('Bot: ' + default_response(user_response))
                
            elif(user_response=='thanks' or user_response=='thank you' ):
                flag=False
                print("Bot:  You are welcome..Don't hesitate to query me again!")
            
            else:
                response = find_response(user_response)
                print(str(response))
                #response.remove(user_response)
        else:
            flag=False
            print("Bot: Bye! take care..")
    
## Main call    
if __name__ == "__main__":
    main()
