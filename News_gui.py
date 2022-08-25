# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 19:12:09 2021

@author: vishnu
"""

# Core Packages
import math
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords  
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter.scrolledtext import *
import tkinter.filedialog
import pickle
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import requests 
import pandas as pd 
from bs4 import BeautifulSoup 

 # Structure and Layout
window = Tk()
window.title("Summaryzer GUI")
window.geometry("700x400")
window.config(background='black')

style = ttk.Style(window)
style.configure('lefttab.TNotebook', tabposition='wn',)


# TAB LAYOUT
tab_control = ttk.Notebook(window,style='lefttab.TNotebook')
 
tab1 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)

# ADD TABS TO NOTEBOOK
tab_control.add(tab1, text=f'{"Home":^20s}')
tab_control.add(tab3, text=f'{"URL":^20s}')


label1 = Label(tab1, text= 'Summaryzer',padx=5, pady=5)
label1.grid(column=0, row=0)

label3 = Label(tab3, text= 'URL',padx=5, pady=5)
label3.grid(column=0, row=0)


tab_control.pack(expand=1, fill='both')

# Functions 
def _create_frequency_matrix(sentences):
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps =  WordNetLemmatizer()

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.lemmatize(word)
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table

    return frequency_matrix

def _create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix

def _create_documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table

def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix

def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix

def _score_sentences(tf_idf_matrix) -> dict:
    """
    score a sentence by its word's TF
    Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue

def _find_average_score(sentenceValue) -> int:
    """
    Find the average score from the sentence value dictionary
    :rtype: int
    """
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original summary_text
    average = (sumValues / len(sentenceValue))

    return average

def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary

def get_summary():
    text=str(entry.get('1.0',tk.END))
	# 2 Create the Frequency matrix of the words in each sentence.
    sentences = sent_tokenize(text) # NLTK function
    total_documents = len(sentences)
    freq_matrix = _create_frequency_matrix(sentences)
    #print(freq_matrix)
    
    '''
    Term frequency (TF) is how often a word appears in a document, divided by how many words are there in a document.
    '''
    # 3 Calculate TermFrequency and generate a matrix
    tf_matrix = _create_tf_matrix(freq_matrix)
    #print(tf_matrix)
    
    # 4 creating table for documents per words
    count_doc_per_words = _create_documents_per_words(freq_matrix)
    #print(count_doc_per_words)
    
    '''
    Inverse document frequency (IDF) is how unique or rare a word is.
    '''
    # 5 Calculate IDF and generate a matrix
    idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
    #print(idf_matrix)
    
    # 6 Calculate TF-IDF and generate a matrix
    tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
    #print(tf_idf_matrix)
    
    # 7 Important Algorithm: score the sentences
    sentence_scores = _score_sentences(tf_idf_matrix)
    #print(sentence_scores)
    
    # 8 Find the threshold
    threshold = _find_average_score(sentence_scores)
    #print(threshold)
    
    # 9 Important Algorithm: Generate the summary
    summary = _generate_summary(sentences, sentence_scores, 1.1 * threshold) 
    result = '\nSummary:{}'.format(summary) 
    tab1_display.insert(tk.END,result)


# Clear entry widget
def clear_text():
	entry.delete('1.0',END)

def clear_display_result():
	tab1_display.delete('1.0',END)


# Clear Text  with position 1.0
def clear_text_file():
	displayed_file.delete('1.0',END)

# Clear Result of Functions
def clear_text_result():
	tab2_display_text.delete('1.0',END)

# Clear For URL
def clear_url_entry():
	url_entry.delete(0,END)

def clear_url_display():
    tab3_display_text.delete('1.0',END)
    url_display.delete('1.0',END)


# Clear entry widget
def clear_compare_text():
	entry1.delete('1.0',END)

def clear_compare_display_result():
	tab1_display.delete('1.0',END)

# Fetch Text From Url
def get_url_summary():
    raw_text = url_display.get('1.0',tk.END)
    sentences=sent_tokenize(raw_text)
    total_documents = len(sentences)
    print(total_documents)
    freq_matrix = _create_frequency_matrix(sentences)
    #print(freq_matrix)
    
    '''
    Term frequency (TF) is how often a word appears in a document, divided by how many words are there in a document.
    '''
    # 3 Calculate TermFrequency and generate a matrix
    tf_matrix = _create_tf_matrix(freq_matrix)
    #print(tf_matrix)
    
    # 4 creating table for documents per words
    count_doc_per_words = _create_documents_per_words(freq_matrix)
    #print(count_doc_per_words)
    
    '''
    Inverse document frequency (IDF) is how unique or rare a word is.
    '''
    # 5 Calculate IDF and generate a matrix
    idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
    #print(idf_matrix)
    
    # 6 Calculate TF-IDF and generate a matrix
    tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
    #print(tf_idf_matrix)
    
    # 7 Important Algorithm: score the sentences
    sentence_scores = _score_sentences(tf_idf_matrix)
    #print(sentence_scores)
    
    # 8 Find the threshold
    threshold = _find_average_score(sentence_scores)
    #print(threshold)
    
    # 9 Important Algorithm: Generate the summary
    summary = _generate_summary(sentences, sentence_scores, 1.1 * threshold) 
    result = '\nSummary:{}'.format(summary)
    tab3_display_text.insert(tk.END,result)	
    
# link for extract html data 
def getdata(): 
    r = requests.get(url_entry.get()) 
    htmldata = r.text
    soup = BeautifulSoup(htmldata, 'html.parser') 
    data = '' 
    text=''
    for data in soup.find_all("p"): 
        text+=''+str(data.get_text())
    
    url_display.insert(tk.END,text)



def preprocessing(text):
    final=[]
    stopWords = set(stopwords.words("english"))
    ps = WordNetLemmatizer()
    s=[]
    sentences = sent_tokenize(text)
    t=[]
    for sent in sentences:
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.lemmatize(word)
            if word in stopWords:
                continue
            else:
                t.append(word)
                
    return t
    
def classify():
    text=str(entry.get('1.0',tk.END))
    categories=['business','entertainment','politics','sport','tech']
    text=preprocessing(text)
    print(text)
    pick=open('data.sav','rb')
    final=pickle.load(pick)
    pick.close()
    cat=[]
    con=[]
    for label,f in final:
        cat.append(label)
        con.append(f)
        
    Tfidf_vect = TfidfVectorizer(max_features=5000)
    text = [str(text)]
    Tfidf_vect.fit(con)
    text_Tfidf = Tfidf_vect.transform(text)
    print(text_Tfidf)
    pick=open('svm.sav','rb')
    model=pickle.load(pick)
    prediction=model.predict(text_Tfidf)
    print(prediction)
    pick.close()
    result = '\nClassification:{}'.format(categories[prediction[0]]) 
    tab1_display.insert(tk.END,result)
    
def get_url_classify():
    text=str(url_display.get('1.0',tk.END))
    categories=['business','entertainment','politics','sport','tech']
    text=preprocessing(text)
    print(text)
    pick=open('data.sav','rb')
    final=pickle.load(pick)
    pick.close()
    cat=[]
    con=[]
    for label,f in final:
        cat.append(label)
        con.append(f)
        
    Tfidf_vect = TfidfVectorizer(max_features=5000)
    text = [str(text)]
    Tfidf_vect.fit(con)
    text_Tfidf = Tfidf_vect.transform(text)
    print(text_Tfidf)
    pick=open('svm.sav','rb')
    model=pickle.load(pick)
    prediction=model.predict(text_Tfidf)
    print(prediction)
    pick.close()
    result = '\nClassification:{}'.format(categories[prediction[0]]) 
    tab3_display_text.insert(tk.END,result)

# MAIN NLP TAB
l1=Label(tab1,text="Enter Text To Summarize")
l1.grid(row=1,column=0)

entry=Text(tab1,height=10)
entry.grid(row=2,column=0,columnspan=2,padx=5,pady=5)

# BUTTONS
button1=Button(tab1,text="Reset",command=clear_text, width=12,bg='#03A9F4',fg='#fff')
button1.grid(row=4,column=0,padx=10,pady=10)

button2=Button(tab1,text="Summarize",command=get_summary, width=12,bg='#03A9F4',fg='#fff')
button2.grid(row=4,column=1,padx=10,pady=10)

button3=Button(tab1,text="Clear Result", command=clear_display_result,width=12,bg='#03A9F4',fg='#fff')
button3.grid(row=5,column=0,padx=10,pady=10)

button4=Button(tab1,text="Classify", command=classify,width=12,bg='#03A9F4',fg='#fff')
button4.grid(row=5,column=1,padx=10,pady=10)
# Display Screen For Result
tab1_display = Text(tab1)
tab1_display.grid(row=7,column=0, columnspan=3,padx=5,pady=5)


# URL TAB
l1=Label(tab3,text="Enter URL To Summarize")
l1.grid(row=1,column=0)

raw_entry=StringVar()
url_entry=Entry(tab3,textvariable=raw_entry,width=50)
url_entry.grid(row=1,column=1)

# BUTTONS
button1=Button(tab3,text="Reset",command=clear_url_entry, width=12,bg='#03A9F4',fg='#fff')
button1.grid(row=4,column=0,padx=10,pady=10)

button2=Button(tab3,text="get text",command=getdata, width=12,bg='#03A9F4',fg='#fff')
button2.grid(row=4,column=1,padx=10,pady=10)

button3=Button(tab3,text="Clear Result", command=clear_url_display,width=12,bg='#03A9F4',fg='#fff')
button3.grid(row=5,column=0,padx=10,pady=10)

button4=Button(tab3,text="Summarize",command=get_url_summary, width=12,bg='#03A9F4',fg='#fff')
button4.grid(row=5,column=1,padx=10,pady=10)

button5=Button(tab3,text="Classify",command=get_url_classify, width=12,bg='#03A9F4',fg='#fff')
button5.grid(row=6,column=0,padx=10,pady=10)

# Display Screen For Result
url_display = ScrolledText(tab3,height=10)
url_display.grid(row=7,column=0, columnspan=3,padx=5,pady=5)


tab3_display_text = ScrolledText(tab3,height=10)
tab3_display_text.grid(row=10,column=0, columnspan=3,padx=5,pady=5)

window.mainloop()