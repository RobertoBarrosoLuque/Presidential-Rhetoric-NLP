import pandas as pd
import numpy as np
import sklearn
import spacy
import matplotlib
import matplotlib.pyplot as plt
import sklearn.feature_extraction.text
from spacy.lang.en.stop_words import STOP_WORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nlp = spacy.load("en_core_web_sm")

def clean_raw_text(raw_texts):
    """
    Clean text documents during pre-processing.
    :param raw_texts: list of raw texts to pre process.
    """

    common_stopwords = ["THE PRESIDENT:", "(Applause.)", "(applause)", "(Laughter.)"]
    stopwords = [x.lower() for x in common_stopwords]
    
    clean_texts = []
    for text in raw_texts:
        try:
            clean_text = text.replace(" \'m", 
                                    "'m").replace(" \'ll", 
                                    "'ll").replace(" \'re", 
                                    "'re").replace(" \'s",
                                    "'s").replace(" \'re", 
                                    "'re").replace(" n\'t", 
                                    "n't").replace(" \'ve", 
                                    "'ve").replace(" /'d", 
                                    "'d").replace('\n','')
            
            clean_text = clean_text.rstrip(" ").rstrip(" ' ").replace("\xa0", "")
            querywords = clean_text.split()
            resultwords  = [word for word in querywords if word.lower() not in stopwords]
            final_text = ' '.join(resultwords)

            clean_texts.append(final_text)
        except AttributeError:
            print("ERROR CLEANING")
            # print(text)
            continue
    return clean_texts


def wordCounter(wordLst):
    """
    Update wordCounter function to avoid counting stopwords.
    :param wordList: list of str words
    """
    wordCounts = {}
    for word in wordLst:
        #We usually need to normalize the case
        wLower = word.lower()
        if wLower in STOP_WORDS:
            continue
        if wLower in wordCounts:
            wordCounts[wLower] += 1
        else:
            wordCounts[wLower] = 1
    #convert to DataFrame
    countsForFrame = {'word' : [], 'count' : []}
    for w, c in wordCounts.items():
        countsForFrame['word'].append(w)
        countsForFrame['count'].append(c)

    result = pd.DataFrame(countsForFrame)
    result.sort_values('count', ascending=False, inplace=True)

    return result

    
def vectorize_documents(dataframe, column, vector_type="tfidf"):
    """
    Wrapper function to vectorize documents.
    :param dataframe: pd.DataFrame
    :param column: name of column in dataframe to vectorize
    """
    if  vector_type == "tfidf":
        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(max_df=100, min_df=2, stop_words='english', norm='l2')
    elif vector_type == "count":
        vectorizer = sklearn.feature_extraction.text.CountVectorizer(max_df=100, min_df=2, stop_words='english', norm='l2')
    else:
        print("Incorrect vector_type input")
        return

    vectorized = vectorizer.fit_transform(dataframe[column])

    return vectorized


def plotSilhouette_refined(n_clusters, num_p_components, vectorized_text):
    """
    A refined version of the plotSilhoette function. Uses dimensionality reduction
    to reduce data in function's body.

    :param n_clusters: int,  k number of clusters
    :num_p_components: int,  number of principal components for PCA
    :vectorized_text: sparse matrix, vectorized form of documents
    """

    X = vectorized_text.toarray()
    PCA = sklearn.decomposition.PCA
    pca = PCA(n_components = num_p_components).fit(vectorized_text.toarray())
    reduced_data = pca.transform(vectorized_text.toarray())

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize = (15,5))
    
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    clusterer = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    
    silhouette_avg = sklearn.metrics.silhouette_score(X, cluster_labels)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = sklearn.metrics.silhouette_samples(X, cluster_labels)

    y_lower = 10
    
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        cmap = matplotlib.cm.get_cmap("nipy_spectral")
        color = cmap(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10
    
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    print(f"Average silhouette score for {n_clusters} clusters is {silhouette_avg}")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    cmap = matplotlib.cm.get_cmap("nipy_spectral")
    colors = cmap(float(i) / n_clusters)
    ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors)

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    projected_centers = pca.transform(centers)
    # Draw white circles at cluster centers
    ax2.scatter(projected_centers[:, 0], projected_centers[:, 1],
                marker='o', c="white", alpha=1, s=200)

    for i, c in enumerate(projected_centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("PC 1")
    ax2.set_ylabel("PC 2")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    plt.show()
    print("For n_clusters = {}, The average silhouette_score is : {:.3f}".format(n_clusters, silhouette_avg))


def modifier(word, POS_tag, df):
    """
    Function to find the most common modifiers for a word belonging to a particular
    part of speach.
    :param word: string, word
    :param POS_tag: string, POS tag of interest
    :param df: pd.DataFrame, df holding POS_senteces column
    """

    NTarget = POS_tag
    NResults = set()
    for entry in df['POS_sentences']:
        for sentence in entry:
            for (ent1, kind1),(ent2,kind2) in zip(sentence[:-1], sentence[1:]):
                if (kind1,ent2.lower())==(NTarget,word):
                    NResults.add(ent1)
                else:
                    continue
    return NResults


def count_pos_tag(df, POS_tag, max_words=None):
    """
    Function to count the number of occurences of a word belong to a particular part of speach.
    :param df: pd.DataFrame, df holding POS_senteces column
    :param POS_tag: string, POS tag of interest
    :param max_words:int, number of words to show
    :return sortedTrgets: list of tuples with most frequent tokens 
    """
    countTarget = POS_tag
    targetCounts = {}
    for entry in df['POS_sentences']:
        for sentence in entry:
            for ent, kind in sentence:
                if kind != countTarget:
                    continue
                elif ent in targetCounts:
                    targetCounts[ent] += 1
                else:
                    targetCounts[ent] = 1
    sortedTargets = sorted(targetCounts.items(), key = lambda x: x[1], reverse = True)
    if max_words:
         sortedTargets = sortedTargets[:max_words]
    return sortedTargets[:max_words]


def count_ner_typew(df):
    """
    Function to count all unique named entity types.
    :param df: dataframe with NER tagged senteces
    :return ner_df: dataframe with counts for all unique ner types
    """
    count_dict = {}
    for cl_sents in df['classified_sents'].values:
        joined = sum(cl_sents, [])
        for ne, ne_type in joined:
            count_dict[ne_type] = count_dict.get(ne_type, 0) + 1

    ner_df = pd.DataFrame(list(count_dict.items()),columns = ["ner_type", "count"]) 
    ner_df.sort_values(by=["count"], ascending=False, inplace=True)

    return ner_df

def find_most_common_ner(n_type, df):
    """
    Find most common NER based on type.
    :param n_type: type of named entity ex: GPE
    :param df: dataframe containing classified sentences
    :return ner_df: dataframe with most common ner type
    """

    count_dict = {}
    for cl_sents in df['classified_sents'].values:
        joined = sum(cl_sents, [])
        for ne, netype in joined:
            if netype == n_type:
                count_dict[ne] = count_dict.get(ne, 0) + 1

    ner_df = pd.DataFrame(list(count_dict.items()),columns = ["ne_name", "count"]) 
    ner_df["type"] = n_type
    ner_df.sort_values(by=["count"], ascending=False, inplace=True)

    return ner_df

def get_depth(root):
    """
    Recursively find max depth of the dependency parse of a spacy chunk by starting with its root
    Code adapted from https://stackoverflow.com/questions/35920826/how-to-find-height-for-non-binary-tree
    :param root: spacy.tokens.token.Token
    :return: int, maximum height of sentence's dependency parse tree
    """
    if not list(root.children):
        return 1
    else:
        return 1 + max(get_depth(x) for x in root.children)


# Find the max depth of each sentence dependency tree
def get_tree_depths(df, sentence_col):
    """
    Calculate the max depth of eache sentence dependency tree.
    :param df: pd.DataFrame with sentence column
    :param sentence_col: column name with tokenized sentences
    :return df:
    """

    all_sentences = []
    depths = []
    for sentence in df.sentence_col.values[0]:
        text = ' '.join(sentence)
        doc = nlp(text)
        max_depths = []
        for chunk in doc.noun_chunks:
            max_depths.append(get_depth(chunk.root))
        all_sentences.append(text)

        if not max_depths:
            max_depths.append(0)

        depths.append(max(max_depths))

    depth_tree = pd.DataFrame({"sentence":all_sentences, "max_depth": depths})
    depth_tree.sort_values(by=["max_depth"], ascending=False, inplace=True)

    return depth_tree


def dep_tree_method(sentence: str = None) -> dict:
    """
    This function will make use of spaCy's dependency tree to identify named
    entities (NEs) and find the sentiment towards such NEs based on the entire
    phrase (or subtree) connected to each NE.

    entity_dct is a dictionary of form:
    {NE1: {'index': i, 'type': ne_type, 'sentiment': sentiment_val},
        NE2: {....}, ... , NE_n: {....}}

    :param sentence: string with sentence.
    :return entity_dct: dictionary with sentiment towards named entities.
    """
    SentimentScorer = SentimentIntensityAnalyzer()
    # analyzer = SentimentIntensityAnalyzer()
    # entity_dct = {'entity_name':{"agg_sentiment":score, "count_subtrees": count, "type": ne_type}}
    entity_dct = {}

    # Check object initialized
    doc_obj = nlp(sentence)

    # get all subtrees in sentence.
    all_subtrees = [[t.text for t in token.subtree] for token in doc_obj]

    # get all entities in sentence.
    entities = [ent.text for ent in doc_obj.ents if ent.label_]
    # entity_cumulative = {'entity_name':{"agg_sentiment":score, "count_subtrees": count}}
    entity_cumulative = {}
    for entity in doc_obj.ents:
        entity_key = entity.text
        entity_cumulative[entity_key] = entity_dct.get(entity.text,
                                                        {"agg_sentiment": 0.0,
                                                        "count_subtrees": 0,
                                                        "type": entity.label_})
        for subtree in all_subtrees:
            # if there is an entity in this subtree calculate a sentiment score and add.
            subtree_text = " ".join(subtree)
            if entity.text in subtree and entity.text != subtree_text:
                # Penalize subtrees which contain more than one entity with penalty = 1 / num_entities_in_subtree
                penalty = 1 / occurrence_counter(subtree_text, entities)
                entity_cumulative[entity_key]["agg_sentiment"] += (SentimentScorer.polarity_scores(subtree_text)["compound"]
                                                                    * penalty)
                entity_cumulative[entity_key]["count_subtrees"] += 1

        # if count_subtrees == 0 then change count_subtrees to 1 to avoid division by 0
        if entity_cumulative[entity_key]["count_subtrees"] == 0:
            entity_cumulative[entity_key]["count_subtrees"] = 1

    entity_dct = {key: {"type": val["type"], "sentiment": round(val["agg_sentiment"] / val["count_subtrees"], 1)}
                    for key, val in entity_cumulative.items()}
    return entity_dct


def occurrence_counter(target_string: str, string_list: list) -> int:
    """
    Helper function to count number of strings in string_list that appear in target_string.
    Used to check number of strings in list that occur in a sentence.
    :param target_string: long string
    :param string_list: list of strings that might appear in target_string
    :return: integer for number of strings in string_list that occur in target_string
    """
    return sum([1 if string in target_string else 0 for string in string_list])