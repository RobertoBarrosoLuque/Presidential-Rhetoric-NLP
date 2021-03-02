import pandas as pd


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


