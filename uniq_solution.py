import preprocess as pr
pr.initStopWords('stopwords')
pr.stemWords(pr.removeStopwords(pr.tokenizeText(open('held_out_tweets.txt').read())))
