# ContentAnalysisPresidentialRhetoric

## Abstract

Our study explores the semantic relationship between presidential speeches and executive orders. Specifically, we are interested in the topics that emerge in speeches and orders, how word use changes over time, and whether there are semantic distinctions between the Republican and Democratic parties. We examine executive orders from the start of Bill Clinton’s presidency up until March 2021 and speeches ranging from Lyndon B. Johnson’s time in office to the beginning of Joe Biden’s presidency. We employ natural language processing techniques such as word frequency analysis, dependency parsing, k-means clustering, topic modeling with Latent Dirichlet Allocation (LDA), word embeddings, and doc2vec. We found that the number of unique words used by each president was mainly a function of how many speeches they gave and the average sentence depth produced by dependency parsing of speeches ranged from 2-6. The distribution of sentiment of presidential speeches is correlated with the general state of the country during each presidential term, with more negatively valenced speeches occurring in presidencies overlapping with war or national crisis. We found that k-means clustering was able to distinguish speeches from orders with high accuracy. We performed topic modeling on the two corpuses using a grid search of LDA models. The relationship between topics of executive orders and of speeches varies based on the president: Barack Obama and George W. Bush appear to exhibit the most consistency between speeches and orders. As expected, executive orders exhibit higher degrees of cosine similarity to one another than do presidential speeches, which have more variation. Word and document embeddings revealed partisan differences in conventionally “divisive” topics such as immigration and climate change. Looking at these embeddings over time reveals major shifts at the start of the Trump presidency as well as the start of the COVID-19 pandemic. Our work extends previous analyses of presidential rhetoric by exploring the relationship between communication and tangible political action. This work aligns with the distributional hypothesis of language, which suggests that words that occur in similar contexts similar meanings and that differences in linguistic form connote differences in meaning. 


## Authors

Lily Grier

Linh Dinh

Roberto Barroso Luque

## Corpora

https://data.world/brianray/c-span-inaugural-address

https://digital.lib.hkbu.edu.hk/corpus/

https://millercenter.org/the-presidency/presidential-speeches

## Structure of Repository
`data`: contains `full_exec_orders_text.csv` and `presidential_speeches.xlsx`, our cleaned corpora

`exploratory_notebooks`:
+ `exec_orders_analysis.ipynb`: all analysis of executive orders
+ `k-means_clutsering-AND-word-embeddings-overtime.ipynb`: k-means clustering of speeches and analysis of word embeddings over time
+ `speeches-semantic-analysis.ipynb`: dependency parsing and word frequency analysis of speeches
+ `speeches-topics-ebmeddings.ipynb`: topic modeling for speeches

`paper_folder`: contains figures and final paper
+ `CCA_speeches.pdf` contains the final paper in PDF form

`scrape_speeches`: contains code for scraping presidential speeches from the Miller corpus

`scrape_orders`: contains code for scraping executive orders from the Federal Register

`topic_modeling`:
+ `PrepareText.py`: contains functions for text cleaning and pre-processing
+ `TopicModeling.py`: contains functions for performing topic modeling using LDA, conducting grid searches, and generating visualizations

`utils`:
+ `helper_functions.py`: contains helper functions for performing clustering and creating visualizations
+ `utils.py`: contains helper functions for pre-processing text as well as dependency parsing and related analysis
