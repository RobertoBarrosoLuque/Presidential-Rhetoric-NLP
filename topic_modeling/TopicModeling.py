import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import gensim
from gensim.models import Phrases
from gensim.models.coherencemodel import CoherenceModel
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
from collections import Counter


class TopicModels:
    """
    Object implementation of topic modeling algorithm (LDA).
    See initializer for more details.
    """

    def __init__(self, prepared_text):
        """
        Initialize TopicModels object.

        Attributes:
        - clean_text: preprocess and cleaned list of documents
        - raw_text: raw_text used before pre processing (list of documents)
        - dictionary: vectorized form of all documents
        - corpus: vectorized form of documents after applying bag of words
                  (t_id, t_freq)
        - corpus_tfidf: vectorized form of documents using TfIDF vectorizer
                  (t_id, t_tf_idf)
        - best_model: gensim model object with highest coherence score.
                      Initialized as None and updated each time
                      optimize_coherence_score or grid_search are run

        :param prepared_text: PreparedText object
        :type prepared_text: PreparedText
        """
        self.clean_text = prepared_text.clean_text
        self.raw_text = prepared_text.raw_text
        self.dictionary = prepared_text.dictionary
        self.corpus = prepared_text.corpus
        self.corpus_tfidf = prepared_text.tfidf_corpus
        self.best_model = None

    def applyLDA(self, numtopics):
        """
        Apply Latent Dirichlet Allocation algorithm to pre processed
        list of documents.
        For more details on LDA see:
        http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf

        :param numtopics: predetermined number of topics for LDA model
        :type numtopics: int
        :return: ldamodel
        :rtype: gensim.models.LdaMulticore
        """
        # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics,
        # id2word=dictionary, passes=15)
        # set random seed for reproducible results
        np.random.seed(1)
        ldamodel = gensim.models.LdaMulticore(corpus=self.corpus_tfidf,
                                              id2word=self.dictionary,
                                              num_topics=numtopics,
                                              alpha='asymmetric',
                                              chunksize=100,
                                              passes=10,
                                              per_word_topics=True)
        return ldamodel

    def gen_coherence_mod(self, ldamodel):
        """
        Generate coherence model.
        Wrapper function for gensim for more info:
        https://radimrehurek.com/gensim/models/coherencemodel.html

        :param ldamodel: gensim topic model instance.
        :type ldamodel: LdaMulticore
        :return: gensim coherence model instance.
        :rtype: CoherenceModel
        """
        coherencemodel = CoherenceModel(model=ldamodel, texts=self.clean_text,
                                        dictionary=self.dictionary,
                                        coherence='c_v')
        return coherencemodel

    def optimize_coherence_score(self, range_numtopics):
        """
        Compute coherence score for multiple lda models by testing different
        number of topics.

        :param range_numtopics: list or numpy array with all values of t number of
                                topics to train on.
        :type range_numtopics: list, np.array
        :return: dataframe for all models trained. Columns are: [model, num_topics, coherence_score]
        :rtype: pd.DataFrame
        """
        coherence_values = []
        model_list = []
        top_coh_val = 0.0
        for numtopics in range_numtopics:
            model = self.applyLDA(numtopics)
            model_list.append(model)

            coherencemodel = self.gen_coherence_mod(model)
            coh_score = coherencemodel.get_coherence()
            if coh_score > top_coh_val:
                top_coh_val = coh_score
                self.best_model = model

            coherence_values.append(coh_score)

        results_df = pd.DataFrame({'Models': model_list,
                                   'num_topics': range_numtopics,
                                   'coherence_score': coherence_values})
        results_df.sort_values(by='coherence_score',
                               ascending=False, inplace=True)

        return results_df

    def grid_search(self, params_grid):
        """
        Manual grid search function to optimize hyperparameters alpha
        (Document topic density), beta (Word-topic density)
        and k (number of topics). Evaluation metric used is a coherence
        score measure.

        :param params_grid: dictionary with parameters to iterate over.
                            i.e. {alpha_range:np.arange(.2, 1, .2),
                            beta_range:np.arange(.2, 1, .2),
                            num_topics_range: [5,1,15,20]}
        :type params_grid: dict
        :return: dataframe with results of grid search
                 ordered in by coherence score (descending)
        :rtype: pd.DataFrame
        """
        coherence_values, model_list = [], []
        alphas_list, betas_list, numtopicslist = [], [], []

        # asymmetric alpha normally yields better results as
        # described here http://dirichlet.net/pdf/wallach09rethinking.pdf
        params_grid['alpha_range'].append('asymmetric')
        params_grid['beta_range'].append('symmetric')

        for a in params_grid['alpha_range']:
            for b in params_grid['beta_range']:
                for k in params_grid['num_topics_range']:
                    ldamodel = gensim.models.LdaMulticore(corpus=self.corpus,
                                                          id2word=self.dictionary,
                                                          num_topics=k,
                                                          chunksize=100,
                                                          passes=10,
                                                          alpha=a,
                                                          eta=b,
                                                          per_word_topics=True)
                    model_list.append(ldamodel)

                    cohmod = self.gen_coherence_mod(ldamodel)
                    coherence_values.append(cohmod.get_coherence())
                    alphas_list.append(a)
                    betas_list.append(b)
                    numtopicslist.append(k)

        results_df = pd.DataFrame({'Models': model_list, 'alpha': alphas_list,
                                   'beta': betas_list, 'num_topics': numtopicslist,
                                   'coherence_score': coherence_values})
        results_df.sort_values(by='coherence_score', ascending=False,
                               inplace=True)
        self.best_model = results_df.iloc[0, 0]

        return results_df

    def get_most_rep_doc(self, ldamodel=None, threshold=.95):
        """
        Create dictionary with the most representative documents
        for each topic
        Where representative is defined as the documents that are at least
        (X) threshold percent related to that topic.
        References:
        https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

        :param ldamodel: trained gensim ldamodel object.
        :type ldamodel: LdaMulticore
        :param threshold: percent doc must be related to topic to be considered representative
        :type threshold: float
        :return: dictionary with {Topic_id: [docindx1, docindx2, .....]}
        :rtype: dict
        """
        # model[corpus] will return a transformed corpus with the first 
        # element as a list of the form [(topic id, percent related), ... ] 
        # this element can be sorted to find which
        # topic is most representative of each document.

        if ldamodel is None:
            ldamodel = self.best_model

        most_rep_docs = {topic_id: [] for topic_id, description
                         in ldamodel.print_topics(-1)}
        for doc_id, doc_topics in enumerate(
                ldamodel.get_document_topics(bow=self.corpus,
                                             per_word_topics=True)):

            # Check to see if document topics representation is empty
            if not doc_topics[0]:
                continue

            topic_num, prop_topic = max(doc_topics[0], key=lambda i: i[1])

            # Check if this document is above threshold for topic
            if threshold < prop_topic:
                most_rep_docs[topic_num].append(doc_id)

        return most_rep_docs

    def create_sum_table(self, ldamodel=None, threshold=.50):
        """
        Function to create pandas df summary table for results of ldamodel
        topic model. Documents include in each topic are only those that meet
        the threshold set in get_most_rep_doc. If there is an empty list for a
        specific topic it means there was no document that was at least X%
        made up of that topic. (Here X represent the threshold set)

        :param ldamodel: fitted gensim ldamodel object
        :type ldamodel: LdaMulticore
        :param threshold: threshold to count a document as part of a topic
        :return:  dataframe with columns:
                  ['Topic_id', 'Keywords', 'number of documents', 'documents ids',  'coherence_val']
        :rtype: pd.DataFrame
        """
        if ldamodel is None:
            ldamodel = self.best_model

        most_rep_docs = self.get_most_rep_doc(ldamodel, threshold=.40)
        results_df = pd.DataFrame()

        for topic_id, _ in ldamodel.show_topics(-1):
            wp = ldamodel.show_topic(topic_id)
            topic_keywords = [(word, prop) for word, prop in wp]
            results_df = results_df.append(pd.Series([str(topic_id + 1),
                                                      topic_keywords, len(most_rep_docs[topic_id]),
                                                      most_rep_docs[topic_id]]), ignore_index=True)
        results_df.columns = ['Topic_id', 'Keywords', 'number of documents', 'Document_indexes']

        # add coherence values for each topic
        coherencemodel = self.gen_coherence_mod(ldamodel)
        coh_list = coherencemodel.get_coherence_per_topic()

        results_df.loc[:, 'coherence_val'] = coh_list

        return results_df

    def convert_topic_dist_df(self, ldamodel=None):
        """
        Create dataframe with rows as documents and columns as topics
        each row contains the distribution of topics for document i,
        adapted from notebook from CCA class at UChicago.txt

        :param ldamodel: fitted gensim ldamodel object
        :type ldamodel: LdaMulticore
        """
        ldaDF = pd.DataFrame()
        #Dict to temporally hold the probabilities
        topicsProbDict = {i : [0] * len(self.raw_text) for i in range(ldamodel.num_topics)}

        #Load them into the dict
        for index, topicTuples in enumerate(ldamodel.get_document_topics(bow=self.corpus)):
            for topicNum, prob in topicTuples:
                topicsProbDict[topicNum][index] = prob

        #Update the DataFrame
        for topicNum in range(ldamodel.num_topics):
            ldaDF['topic_{}'.format(topicNum+1)] = topicsProbDict[topicNum]
        
        return ldaDF

    def generate_topic_wordcloud(self, ldamodel=None):
        """
        Generate wordcloud using most important words for each topic.

        :param ldamodel: fitted gensim ldamodel
        :type ldamodel: LdaMulticore
        """
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

        cloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          width=2500,
                          height=1800,
                          max_words=15,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        if ldamodel is None:
            topics = self.best_model.show_topics(formatted=False)
        else:
            topics = ldamodel.show_topics(formatted=False)

        num_topics = len(topics)
        fig, axes = plt.subplots(num_topics // 2, 2, figsize=(20, 20), sharex='all', sharey='all')

        for i, ax in enumerate(axes.flatten()):
            fig.add_subplot(ax)
            topic_words = dict(topics[i][1])
            cloud.generate_from_frequencies(topic_words, max_font_size=300)
            plt.gca().imshow(cloud)
            plt.gca().set_title('Topic ' + str(i + 1), fontdict=dict(size=16))
            plt.gca().axis('off')
        plt.subplots_adjust(wspace=0, hspace=.3)

    def generate_bar_plot(self, docs_to_view, ldamodel=None, title=None):
        """
        Function to generate bar plot to visualize distribution of topics among documents.
        Code based on notebook from Computational Content Analysis class at UChicago.

        :param docs_to_view: indexes of documents to plot
        :param ldamodel: fitted gensim ldamodel
        :param title: title for plot
        """
        if ldamodel is None:
            ldamodel = self.best_model

        LDAdf = self.convert_topic_dist_df(ldamodel=ldamodel)
        LDAdf = LDAdf.iloc[docs_to_view,:]
        LDAdf.plot(kind="bar", stacked=True)
        plt.legend(loc="right")
        plt.legend(loc=(1.04,0))
        
        if title:
            plt.title(title)


    def __repr__(self):
        return 'This is a LDA topic model object'


def get_topic_words(model):
    """
    Convert cluster word distribution to only include words

    :param model: instance of sttm model object
    :type model: MovieGroupProcess
    :return: list of lists with each topic represented
             by the words it is made up of.
    :rtype: list
    """
    topics = []
    for d in model.cluster_word_distribution:
        if not d:
            continue
        t = [w for w in d.keys()]
        topics.append(t)
    return topics