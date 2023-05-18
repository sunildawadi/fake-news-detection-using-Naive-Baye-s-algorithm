#!/usr/bin/env python
# coding: utf-8

# In[11]:


from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pickle
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from langdetect import detect
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer


# In[12]:


app = Flask(__name__)


# In[13]:


# Load model
try:
    with open('naive_bayes_model.pkl', 'rb') as file:
        model_train = pickle.load(file)
except FileNotFoundError:
    print("Error: unable to find the trained model file named 'naive_bayes_model.pkl'.")
    exit()


# In[14]:


# Load the vectorizer
try:
    with open('tfidf_vectorizer.pkl', 'rb') as file:
        model_vectorizer = pickle.load(file)
except FileNotFoundError:
    print("Error: unable to find the vectorizer file named 'tfidf_vectorizer.pkl'.")
    exit()


# In[15]:


translator = str.maketrans('', '', string.punctuation)


# In[16]:


def preprocess_article(article):
    article = article.lower()
    article = article.translate(translator)
    stop_words = stopwords.words('english')
    article = ' '.join([word for word in article.split() if word not in (stop_words)])
    print(article)
    article = sent_tokenize(article)
    # Tokenize each sentence into words
    article = [word_tokenize(sentence) for sentence in article]
    
    return article

def preprocess_article_tk(article):
    article = article.lower()
    article = article.translate(translator)
    stop_words = stopwords.words('english')
    article = ' '.join([word for word in article.split() if word not in (stop_words)])
    print(article)    
    return article


# In[17]:


def predict_fake_news_naive_bayes(X, prob_fake, prob_real, fake_word_probs, real_word_probs, alpha=1):
    num_fake = 0
    num_real = 0
    y_pred = []
    for i in range(X.shape[0]):
        article = X[i, :]
        prob_real_article = 1.0
        prob_fake_article = 1.0
        words = article.nonzero()[1]
        for index in words:
            if index in fake_word_probs:
                prob_fake_article *= fake_word_probs[index]
            else:
                prob_fake_article *= alpha / (num_fake + 2 * alpha)
            if index in real_word_probs:
                prob_real_article *= real_word_probs[index]
            else:
                prob_real_article *= alpha / (num_real + 2 * alpha)
        
        
        # Predict the class label of the news article
        if prob_fake_article * prob_fake > prob_real_article * prob_real:
            y_pred.append(1)
        else:
            y_pred.append(0)
            
        if y_pred[-1] == 1:
            num_fake += 1
        else:
            num_real += 1
            
    return y_pred




def check_repetitive_words(article):
    word_counts = defaultdict(int)
    repetitive_words = []
    print("aayo")
    for word in article:
            for new_word in word:
                print(new_word)
                word_counts[new_word] += 1

    if word_counts[new_word] > 50:
        repetitive_words.append(1)
    else:
        repetitive_words.append(0)
    print("sucess")
    print(repetitive_words)
    return np.array(repetitive_words)

# In[18]:


#Flask routing 
@app.route("/")

def home():
    return render_template("Home.html")


# In[19]:


@app.route('/predict', methods = ['POST','GET'])
def predict_news():
    if request.method == 'POST':
        news = request.form['message']
        vector_news = news
        try: 
            language = detect(news)
            print(language)  # Output: 'en' (English)
            if language == 'en':
                news = preprocess_article(news)
                print(news)
                print("here")
                rep_word = check_repetitive_words(news)
                print("hello")
                if rep_word == [0]:
                    vector_news = preprocess_article_tk(vector_news)
                    print("aayo yaa")
                    print(vector_news)
                    # Create an instance of TfidfVectorizer
                    #vectorizer = TfidfVectorizer()
                    #vectorizer.fit(news)
                    #vectorized_article = vectorizer.transform(news)
                    vectorized_article = model_vectorizer.transform([vector_news])
                    print("function")
                    prediction = predict_fake_news_naive_bayes(vectorized_article, model_train["prob_fake"], model_train["prob_real"], 
                                        model_train["fake_word_probs"], model_train["real_word_probs"])[0]
                    pred = [prediction]
                    print(pred)
                    return render_template('Home.html', prediction=pred)
                else:
                    return render_template('Home.html', error="Please don't use repetitive words!!")

            else:
                return render_template('Home.html', error="Provide a valid news")
            
        except Exception as e:
            error = str(e)
            return render_template('Home.html', error=error)
    else:
        return render_template('Home.html')


# In[20]:


if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port='9019')


# In[ ]:





# In[ ]:





# In[ ]:




