import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn import metrics, pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem
import warnings
warnings.filterwarnings("ignore")


def load_data(training_path, testing_path):
    # load training and testing data
    df_train = pd.read_csv('training.csv')
    df_test = pd.read_csv('test.csv')
    return df_train, df_test


def pre_processing_data(df_train, df_test, stop_words):
    # remove irrelevant items
    df_train = df_train.loc[df_train['topic'] != 'IRRELEVANT']
    df_test = df_test.loc[df_test['topic'] != 'IRRELEVANT']

    # add topic codes for each topic
    # generate topic codes for each topic
    topic_codes = {
        'ARTS CULTURE ENTERTAINMENT': 0,
        'BIOGRAPHIES PERSONALITIES PEOPLE': 1,
        'DEFENCE': 2,
        'DOMESTIC MARKETS': 3,
        'FOREX MARKETS': 4,
        'HEALTH': 5,
        'MONEY MARKETS': 6,
        'SCIENCE AND TECHNOLOGY': 7,
        'SHARE LISTINGS': 8,
        'SPORTS': 9
    }

    # Category mapping
    df_train['topic_codes'] = df_train['topic']
    df_train = df_train.replace({'topic_codes': topic_codes})
    df_test['topic_codes'] = df_test['topic']
    df_test = df_test.replace({'topic_codes': topic_codes})

    # remove stop words
    df_train['content_parsed'] = df_train['article_words']
    df_test['content_parsed'] = df_test['article_words']
    for stop_word in stop_words:
        regex_stopword = r'\b' + ',' + stop_word + r'\b'
        regex2_stopword = r'\b' + stop_word + ',' + r'\b'
        df_train['content_parsed'] = df_train['content_parsed'].str.replace(regex_stopword, '')
        df_train['content_parsed'] = df_train['content_parsed'].str.replace(regex2_stopword, '')
        df_test['content_parsed'] = df_test['content_parsed'].str.replace(regex_stopword, '')
        df_test['content_parsed'] = df_test['content_parsed'].str.replace(regex2_stopword, '')

    # add article length and id information
    df_train['article_length'] = df_train['content_parsed'].str.len()
    df_test['article_length'] = df_test['content_parsed'].str.len()
    df_train['id'] = 1
    df_test['id'] = 1

    return df_train, df_test


def features_extract(df_train, df_test, nb_features):
    # for each class, get the most frequent  words as feature vectors
    words_set_train = set()
    for i in range(1, 11, 1):
        bag = df_train[df_train['topic_codes'] == i]['content_parsed']
        total_text = ""
        for text in bag:
            total_text += (text + ",")
        temp_set = set(pd.value_counts(total_text.split(","))[0:nb_features].keys())
        words_set_train = words_set_train.union(temp_set)

    words_set_test = set()
    for i in range(1, 11, 1):
        bag = df_test[df_test['topic_codes'] == i]['content_parsed']
        total_text = ""
        for text in bag:
            total_text += (text + ",")
        temp_set = set(pd.value_counts(total_text.split(","))[0:100].keys())
        words_set_test = words_set_test.union(temp_set)

    # get the intersection of two feature words(including training feature words and testing feature words)
    # and use the intersection as our feature vector X
    words_set = words_set_train.intersection(words_set_test)
    words_list = sorted(list(words_set))
    return words_list


def regularize_data(df_train, df_test, words_list):
    # regularize training data
    new_content_train = []
    for row in df_train['content_parsed']:
        temp_row = row.split(",")
        new_row = []
        for word in temp_row:
            if word in words_list:
                new_row.append(word)
        new_str = ",".join(new_row)
        new_content_train.append(new_str)
    df_train['content_parsed_2'] = new_content_train

    # regularize testing data
    new_content_test = []
    for row in df_test['content_parsed']:
        temp_row = row.split(",")
        new_row = []
        for word in temp_row:
            if word in words_list:
                new_row.append(word)
        new_str = ",".join(new_row)
        new_content_test.append(new_str)
    df_test['content_parsed_2'] = new_content_test

    return df_train, df_test


def training_process_for_MultinomialNB_model(X_train, y_train, X_test, y_test):
    # # get prior probability (probability of each topic is not uniformed)
    # prior_pro = dict(df_train.groupby('topic_codes').count()['topic'] / df_train.shape[0])
    # class_prior = list(prior_pro.values())

    # create model and predict result
    clf = MultinomialNB(fit_prior=True, class_prior=None)
    model = clf.fit(X_train, y_train)
    predicted_y = model.predict(X_test)
    print(accuracy_score(y_test, predicted_y))
    print(precision_score(y_test, predicted_y, average='macro'))
    print(recall_score(y_test, predicted_y, average='macro'))
    print(f1_score(y_test, predicted_y, average='macro'))
    print(classification_report(y_test, predicted_y))


def training_process_for_MultinomialLR_model(X_train, y_train, X_test, y_test):
    # create model and predict result
    clf_lr = LogisticRegression(C=1.0, solver='lbfgs', multi_class='multinomial')
    clf_lr.fit(X_train, y_train)
    predicted_y = clf_lr.predict(X_test)
    print(accuracy_score(y_test, predicted_y))
    print(precision_score(y_test, predicted_y, average='macro'))
    print(recall_score(y_test, predicted_y, average='macro'))
    print(f1_score(y_test, predicted_y, average='macro'))
    print(classification_report(y_test, predicted_y))


if __name__ == '__main__':
    # get stopwords
    nltk.download('stopwords')
    stop_words = list(stopwords.words('english'))  # get stop words
    print(stop_words)

    # do some data cleaning and pre-processing
    df_train, df_test = load_data("training.csv", "test.csv")
    df_train, df_test = pre_processing_data(df_train, df_test, stop_words)
    feature_words = features_extract(df_train, df_test, 100)
    df_train, df_test = regularize_data(df_train, df_test, feature_words)

    # create bag of words
    text_data_train = np.array(df_train['content_parsed_2'])
    count_train = CountVectorizer()
    bag_of_words_train = count_train.fit_transform(text_data_train)

    text_data_test = np.array(df_test['content_parsed_2'])
    count_test = CountVectorizer()
    bag_of_words_test = count_test.fit_transform(text_data_test)

    # Create feature matrix and target, train model
    X_train = bag_of_words_train.toarray()
    y_train = np.array(df_train['topic_codes'])
    X_test = bag_of_words_test.toarray()
    y_test = np.array(df_test['topic_codes'])

    training_process_for_MultinomialNB_model(X_train, y_train, X_test, y_test)


    mll_scorer = metrics.make_scorer(multiclass_logloss, greater_is_better=False, needs_proba=True)
    nb_model = MultinomialNB()

    # 创建pipeline
    clf = pipeline.Pipeline([('nb', nb_model)])

    # 搜索参数设置
    param_grid = {'nb__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

    # 网格搜索模型（Grid Search Model）初始化
    model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=mll_scorer,
                         verbose=10, n_jobs=-1, iid=True, refit=True, cv=6)

    # fit网格搜索模型
    model.fit(X_train, y_train)
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

