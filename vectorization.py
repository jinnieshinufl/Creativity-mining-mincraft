from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 

class LemmaTokenizer(object):
        def __init__(self):
            self.wnl = WordNetLemmatizer()
        def __call__(self, articles):
            return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]
        
df_short = df_test 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
corpus = df_short.text.tolist()

vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(),
                                strip_accents = 'unicode', # works 
                                stop_words = 'english', # works
                                lowercase = True, 
                            ngram_range = (0,3))
X = vectorizer.fit_transform(corpus)
#X= feat
#print(X)
#X1= pd.DataFrame(entity_freq).fillna(0)

y = df_short[['Elaboration', 'Originality', 'Aesthetics ',
                     'Surprise',  'Complexity',  #humor removed - no entity 
                     'Realism/ recreation']] #novel removed - 3 entities 
y = y.fillna(0) 
y = y.sum(axis=1) > 0 

#from collections import Counter
#Counter(y)

y = y*1
clf = LogisticRegression()
clf.fit(X, y)
cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=1)
# evaluate the model and collect the scores
n_scores = cross_val_score(clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report the model performance
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
