#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing ,CSV file I/O (e.g. pd.rad_csv)
import string  # special operations on strings
import spacy   # Language models

from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

elon=pd.read_csv("C:/Users/Administrator/Downloads/Elon_musk.csv",encoding='latin',sep=",", 
                 error_bad_lines=False,warn_bad_lines=False)
elon.head()


# In[3]:


elon.drop(['Unnamed: 0'],inplace=True,axis=1)
elon


# # Text Processing

# In[4]:


# Text processing

elon = [Text.strip() for Text in elon['Text']]      # removing the trailing and leading characters
elon = [Text for Text in elon if Text]           # removing the empty strings from the data
elon[0:10]


# In[5]:


# joining the list of comments into a single text/string

text = ' '.join(elon)
text


# In[6]:


len(text)


# In[7]:


#Punctuation

no_punc_text = text.translate(str.maketrans('','',string.punctuation))
no_punc_text


# In[8]:


#Tokenization
import nltk
from nltk.tokenize import word_tokenize  
nltk.download('punkt')
text_tokens=word_tokenize(no_punc_text)
print(text_tokens)


# In[9]:


#Stopwords
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


# In[10]:


len(text_tokens)


# In[11]:


# Remove Stopwords
from nltk.corpus import stopwords
my_stop_words=stopwords.words('english')

sw_list = ['\x92','rt','ye','yeah','haha','Yes','U0001F923','I']
my_stop_words.extend(sw_list)

no_stop_tokens=[word for word in text_tokens if not word in my_stop_words]
print(no_stop_tokens)


# In[12]:


# Normalize the data
lower_words=[Text.lower() for Text in no_stop_tokens]
print(lower_words[100:200])


# In[13]:


# Stemming (Optional)
from nltk.stem import PorterStemmer
ps=PorterStemmer()
stemmed_tokens=[ps.stem(word) for word in lower_words]
print(stemmed_tokens[100:200])


# In[14]:


pip install spacy


# In[15]:


get_ipython().system('python -m spacy download en_core_web_sm')
import spacy
nlp = spacy.load("en_core_web_sm")


# In[16]:


# Lemmatization
nlp=spacy.load('en_core_web_sm')
doc=nlp(' '.join(lower_words))
print(doc)


# In[17]:


lemmas = [token.lemma_ for token in doc]


# In[18]:


clean_tweets=' '.join(lemmas)
clean_tweets


# # Feature Extraction

# In[19]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
tweetscv=cv.fit_transform(lemmas)


# In[20]:


print(cv.vocabulary_)


# In[21]:


print(cv.get_feature_names()[100:200])


# In[22]:


print(tweetscv.toarray()[100:200])


# In[23]:


print(tweetscv.toarray().shape)


# ## Count vectorizer with Bi - gram & Tri - grams

# In[24]:


cv_ngram_range=CountVectorizer(analyzer='word',ngram_range=(1,3),max_features=100)  #Bigrams and Trigrams
bow_matrix_ngram=cv_ngram_range.fit_transform(lemmas)


# In[25]:


print(cv_ngram_range.get_feature_names())
print(bow_matrix_ngram.toarray())


# ## TF-IDF Vectorizer

# In[26]:


from sklearn.feature_extraction.text import TfidfVectorizer    #Tf idf Vectorizer
tfidfv_ngram_max_features=TfidfVectorizer(norm='l2',analyzer='word',ngram_range=(1,3),max_features=500)
tfidf_matix_ngram=tfidfv_ngram_max_features.fit_transform(lemmas)


# In[27]:


print(tfidfv_ngram_max_features.get_feature_names())
print(tfidf_matix_ngram.toarray())


# ## Generate World

# In[28]:


def plot_cloud(wordcloud):             #word cloud
    plt.figure(figsize=(40,30))
    plt.imshow(wordcloud)
    plt.axis('off')
    
# Generate Word Cloud

STOPWORDS.add('pron')
STOPWORDS.add('rt')
STOPWORDS.add('yeah')
wordcloud=WordCloud(width=3000,height=2000,background_color='black',max_words=50,
                   colormap='Set1',stopwords=STOPWORDS).generate(clean_tweets)
plot_cloud(wordcloud)


# ## Named Entity Recognition

# In[29]:


# Parts Of Speech (POS) Tagging
nlp=spacy.load('en_core_web_sm')

one_block=clean_tweets
doc_block=nlp(one_block)
spacy.displacy.render(doc_block,style='ent',jupyter=True)


# In[30]:


for token in doc_block[100:200]:
    print(token,token.pos_)


# In[31]:


# Filtering the nouns and verbs only
nouns_verbs=[token.text for token in doc_block if token.pos_ in ('NOUN','VERB')]
print(nouns_verbs[100:200])


# In[32]:


# Counting the noun & verb tokens
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

X=cv.fit_transform(nouns_verbs)
sum_words=X.sum(axis=0)

words_freq=[(word,sum_words[0,idx]) for word,idx in cv.vocabulary_.items()]
words_freq=sorted(words_freq, key=lambda x: x[1], reverse=True)

wd_df=pd.DataFrame(words_freq)
wd_df.columns=['word','count']
wd_df[0:10] # viewing top ten results


# In[33]:


# Visualizing results (Barchart for top 10 nouns + verbs)
wd_df[0:10].plot.bar(x='word',figsize=(12,8),title='Top 10 nouns and verbs',color = 'blue');


# ## Sentiment analysis

# ### Emotion Mining

# In[34]:


import pandas as pd


# In[35]:


afinn = pd.read_csv("C:/Users/Administrator/Downloads/Afinn.csv",sep=',',encoding='latin-1')
afinn.shape


# In[36]:


afinn.head()


# In[37]:


import numpy as np  # linear algebra
import pandas as pd  # data processing ,CSV file I/O (e.g.read_csv)
import string        # special operations on strings
import spacy         # language models

from matplotlib.pyplot import imread
from matplotlib import pyplot as plt
from wordcloud import WordCloud
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas
book = pd.read_csv("C:/Users/Administrator/Downloads/apple.txt",error_bad_lines=False)
book = [x.strip()for x in book.x] # remove both the leading and the trailing characters
book = [x for x in book if x] # removes empty strings,because they are considered in a python as False


# In[38]:


from nltk import tokenize
sentences = tokenize.sent_tokenize(" ".join(book))
sentences[5:15]


# In[39]:


sent_df = pd.DataFrame(sentences, columns=['sentence'])
sent_df


# In[40]:


affinity_scores = afinn.set_index('word')['value'].to_dict()


# In[41]:


# Custom function :score each word in a sentence in lemmatised form,
#but calculate the score for the whole original sentence.

nlp = spacy.load('en_core_web_sm')
sentiment_lexicon = affinity_scores

def calculate_sentiment(text: str = None):
    sent_score = 0
    if text:
        sentence = nlp(text)
        for word in sentence:
            sent_score += sentiment_lexicon.get(word.lemma_, 0)
    return sent_score


# In[42]:


# test that it works
calculate_sentiment(text = 'amazing')


# In[43]:


sent_df['sentiment_value'] = sent_df['sentence'].apply(calculate_sentiment)


# In[44]:


# how many words are i  the sentence?
sent_df['word_count'] = sent_df['sentence'].str.split().apply(len)
sent_df['word_count'].head(10)


# In[45]:


sent_df.sort_values(by='sentiment_value').tail(10)


# In[46]:


# Sentiment score of the whole review
sent_df['sentiment_value'].describe()


# In[47]:


sent_df[sent_df['sentiment_value']>=20].head()


# In[48]:


sent_df['index']=range(0,len(sent_df))


# In[49]:


import seaborn as sns 
import matplotlib.pyplot as plt 
sns.distplot(sent_df['sentiment_value'])


# In[50]:


plt.figure(figsize=(15,10))
sns.lineplot(y='sentiment_value',x='index',data=sent_df)


# In[51]:


sent_df.plot.scatter(x='word_count',y='sentiment_value',figsize=(8,8),title='Sentence sentiment value to sentence word count')


# In[ ]:




