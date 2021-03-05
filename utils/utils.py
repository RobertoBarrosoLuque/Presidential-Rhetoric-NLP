import pandas as pd
import numpy as np
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import sklearn.feature_extraction.text


def label_presidents_climate():
    """
    Label presidential speeches corpus on individual president's
    attitute towards climate change mitigation.
    """
    speeches = pd.read_excel(r"../data/presidential_speeches.xlsx")
    speeches = speeches.drop(columns="Unnamed: 0")
    speeches.columns
    climate_words = ["climate", "energy", "environment", "global warming",  "fossil fuels", 
                    "pollution", "emissions", "water", "air", "clean", "coal", "oil", "greenhouse gases", "carbon dioxide"
                    ,"methane"]
    speeches = speeches.loc[speeches.President.isin(speeches.President.unique()[:6]), :]

    related = []
    neutral = []
    for i, president, date, speech in speeches.itertuples():
        
        if any(st in speech for st in climate_words):
            related.append(i)
        else:
            neutral.append(i)

    speeches['climate_stance'] = None
    speeches.iloc[speeches.index.isin(neutral), 3] = "neutral"


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