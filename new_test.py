from finacial_sentiment import *
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

text_proc = TextProcessing('data.csv')

def test_my_model():
    over_samp = text_proc.oversample_random(
        text_proc.bag_of_words(text_proc.no_punc_sent))
    train_x, val_x, train_y, val_y = train_test_split(over_samp[0],
                                                      over_samp[1],
                                                      test_size=0.3)

    model = MODELS['SVM']
    model.fit(train_x, train_y)
    pred = model.predict(val_x)
    assert f1_score(val_y, pred, average='weighted') > 0.85

def test_sentiment():
    train_x, val_x, train_y, val_y = train_test_split(text_proc.sent,
                        text_proc.labels, test_size=0.3)

    pipeline_log = Pipeline([
         ('count', CountVectorizer()),
         ('tfidf', TfidfTransformer()),
         ('clf', LogisticRegression(solver='liblinear', multi_class='auto')),
        ])

    model = pipeline_log.fit(train_x, train_y)
    sent = ["they had another fantastic year. \
            Their profits are so high. I am very excited."]
    pred = model.predict(sent)
    assert pred == [1]
