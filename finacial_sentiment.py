import pandas as pd
import nltk
import string
import csv
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#conda install -c glemaitre imbalanced-learn
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
np.random.seed(111)
stopwords = nltk.corpus.stopwords.words('english')
import warnings
warnings.filterwarnings("ignore")

class TextProcessing:
    def __init__(self, file):
        # read the data using pandas
        self.df = pd.read_csv(file)
        self.sent = self.df['Sentence'].astype(str).str.lower()  # text not cleaned
        self.no_punc_sent = self._remove_punc()
        self.cleaned_sent = self._process_text()  # Text cleaned
        self.labels = self._process_labels()

    # remove punctuation only
    def _remove_punc(self):
        text = self.sent.apply(lambda x: ' '.join(
            [word for word in x.split() if word not in string.punctuation]))
        return text

    # process data by lowering, removing stopwords & punctuation
    def _process_text(self):
        sentences = self.sent.apply(lambda x: ' '.join([word for word \
            in x.split() if word not in set(stopwords) and string.punctuation]))
        return sentences

    def tfidf_vectorizer(self, text):  #tf-idf vectorizer
        vectorizer = TfidfVectorizer()
        tf_idf = vectorizer.fit_transform(text).toarray()
        return tf_idf

    def bag_of_words(self, text):  # bag_of_word vectorizer
        vectorizer = CountVectorizer()
        bow = vectorizer.fit_transform(text).toarray()
        return bow

    def _process_labels(self):
        labels = self.df['Sentiment'].apply(lambda x: -1 if \
            x == 'negative' else (1 if x == 'positive' else 0))
        return labels

    # to oversample minority class using SMOTE
    def oversample_smote(self, text, sampling_strategy='auto'):
        oversample = SMOTE(sampling_strategy=sampling_strategy)
        oversample_sent, labels_final = oversample.fit_resample(text, self.labels)
        counter = Counter(labels_final)
        print(counter)
        return oversample_sent, labels_final

    # to randomly oversample by RandomOverSampler
    def oversample_random(self, text):
        oversample = RandomOverSampler()
        oversample_sent, labels_final = oversample.fit_resample(text, self.labels)
        return oversample_sent, labels_final

    # to randomly undersample by RandomUnderSampler
    def under_over_sample(self, text, sampling_strategy):
        x, y = self.oversample_smote(text)
        oversample = RandomUnderSampler(sampling_strategy=sampling_strategy)
        oversample_sent, labels_final = oversample.fit_resample(x,y)
        counter = Counter(labels_final)
        print(counter)
        return oversample_sent, labels_final


# define models with cost sensitive approach and specific hyperparameters
MODELS = {'Logistic Regression': LogisticRegression(multi_class='multinomial',
                      class_weight='balanced', C=10, penalty='l2'),
          'SVM': SVC(class_weight='balanced', C=10),
          'Random Forest': RandomForestClassifier(class_weight='balanced')}

# to select best model based on weighted average F1 score on validation set
def best_model(models, text, labels, measure):
    train_x, val_x, train_y, val_y = train_test_split(text, labels, test_size=0.3)

    for name, model in models.items():
        model.fit(train_x, train_y)
        pred = model.predict(val_x)
        if measure == 'f1':
            print(name, f1_score(val_y, pred, average='weighted'))
        elif measure == 'accuracy':
            print(name, accuracy_score(val_y, pred))
        else:
            print('Please select f1, f1_weighted or accuracy as metric.')

'''
text_proc = TextProcessing('data.csv')

over_samp = text_proc.oversample_random(
    text_proc.bag_of_words(text_proc.no_punc_sent))



for testphrase in sent:
    resultx = model_lgr.predict([testphrase])
    print(resultx)


print("BOW Cleaned - Random")
over_samp = text_proc.oversample_random(text_proc.bag_of_words(text_proc.cleaned_sent))
best_model(MODELS, over_samp[0], over_samp[1], 'f1')

print("TFIDF Cleaned - Random")
comb = text_proc.oversample_random(text_proc.tfidf_vectorizer(text_proc.cleaned_sent))
best_model(MODELS, comb[0], comb[1], 'f1')

print("TFIDF Cleaned - SMOTE")
comb = text_proc.oversample_smote(text_proc.tfidf_vectorizer(text_proc.cleaned_sent))
best_model(MODELS, comb[0], comb[1], 'f1')

print("TFIDF Cleaned - Combination")
comb = text_proc.under_over_sample(text_proc.tfidf_vectorizer(text_proc.cleaned_sent), sampling_strategy)
best_model(MODELS, comb[0], comb[1], 'f1')
'''









