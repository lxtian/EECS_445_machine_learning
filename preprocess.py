#!/usr/bin/python
import sys, re, os
from string import maketrans
import datetime
from porter import PorterStemmer
from random import shuffle
from math import log

GLOBAL = {} # global variables

GLOBAL['folder'] = sys.argv[1] if len(sys.argv) > 1 else "cranfieldDocs/"

def initStopWords(filename):
    '''
    initialize GLOBAL['stopwords'] as defined in
    http://web.eecs.umich.edu/~mihalcea/courses/498IR/Resources/stopwords
    '''
    with open(filename) as f:
        GLOBAL["stopwords"] = set([word.strip() for word in f])

def removeSGML(text):
    ''' 
    Name: removeSGML
    input: string
    output: string
    '''
    return re.sub(r'<.*?>','',text)

def tokenizeText(text, pattern='\s+'):
    ''' 
    return list of tokens
    tokenization of.(donot tokenize acronyms,abbreviations,numbers)
    tokenization of'(expand when needed, e.g.,I'm->I am;
                    tokenize the possessive,
                    e.g.,Sunday's->Sunday 's;etc.)
    tokenization of dates (keep dates together)
    tokenization of - (keep phrases separated by - together)
    '''

    out = []
    punctuation = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
    raw = re.split(pattern,text.strip())

    def acronym(t,i): 
        # naive
        if t == t.capitalize(): return True
        if i-1 >= 0:
            if '.' in raw[i-1]: return True
        if i+1 < len(raw):
            if '.' in raw[i+1]: return True
            if raw[i+1].capitalize() == raw[i+1]:
                return True
        else: return False

    def abbr(t): 
        if len(t) <= 2: return True
        # more than 1 dot
        ndot = 0
        for c in t:
            if c == '.': ndot += 1
        if ndot >= 2 : return True
        return t==t.capitalize()

    def number(t):
        try:
            float(t)
            return True
        except ValueError:
            return False

    def isdate(t):
        # naive implementation
        p1 = re.compile('(\d+)/(\d+)/(\d+)')
        p2 = re.compile('(\d+).(\d+).(\d+)')
        candidate = p1.findall(t) or p2.findall(t)
        if candidate:
            day, month, year = candidate[0]
            if 1 <= day <= 31 and 1 <= month <= 12 and 500 <= year <= 2500:
                return True
            year, day, month = candidate[0]
            if 1 <= day <= 31 and 1 <= month <= 12 and 500 <= year <= 2500:
                return True
        return False
        
    for i, token in enumerate(raw):
        if '.' in token and len(token) > 1 \
                and (acronym(token,i) or abbr(token) or number(token)): pass
        # naive
        elif "'" in token: 
            t_i = token.find("'")
            out.append(token[:t_i])
            token = token[t_i:]
        elif "-" in token: pass
        elif isdate(token): pass
        else: token = token.translate(maketrans('',''),punctuation)
        if token:
            out.append(token)
    return out

def removeStopwords(tokens):
    ''' 
    input: list of tokens
    output: list of tokens with stopwords removed
    refer to here:
    '''
    res = []
    for i, w in enumerate(tokens):
        if w not in GLOBAL['stopwords']:
            res.append(w)
    return res

def stemWords(tokens):
    ''' 
    input: list of tokens
    output: list of stemmed tokens
    use the porter.py
    '''
    p = PorterStemmer()
    return map(lambda t: p.stem(t,0,len(t)-1), tokens)

def test(fn=['cranfield000'+str(i) for i in range(1,10)],p=False):
    tot = []
    for filename in fn:
        with open(folder+filename) as f:
            text = f.read()
            SGMLremoved = removeSGML(text)
            tokenized = tokenizeText(SGMLremoved)
            stop = removeStopwords(tokenized)
            tot += stemWords(stop)
            if p:
                print "filename " + filename
                print "Orignal text:--------------------------------"
                print text + '\n'
                print "SGML removed:----------------------------------------------"
                print SGMLremoved
                print 
                print "Tokenized:-------------------------------------------------"
                print tokenized
                print
                print "stop words removed---------------------------------------------"
                print stop       
                print 
                print "stemmed result--------------------------------------------------"
                print tot
                print 
                print "required output-----------------------------------------------------"
    
    c = {}
    for token in tot:
        c.setdefault(token,0)
        c[token] += 1

    n = 50
    tot_words = sum(c.values()) 
    Vocabulary = len(c.keys())
    print "Words %d" % tot_words
    print "Vocabulary %d" % Vocabulary
    print "Top %d words" % n

    sorted_list = sorted(c,key=c.get,reverse=True)
    for k in sorted_list[:n]:
        print "%15s %10d" % (k, c[k])
        
    proportion = 0.25
    threshold = proportion * tot_words

    curr_tot = 0
    for n_min, key in enumerate(sorted_list):
        curr_tot += c[key]
        if curr_tot > threshold:
            n_min += 1
            break
        
    print "The minimum unique words accounting for %.2f%% of total number of words is %3d" % (proportion,n_min)    

def summary_output_supressed(fn):    
    tot = []
    for filename in fn:
        with open(folder+filename) as f:
            tot += stemWords(removeStopwords(tokenizeText(removeSGML(f.read()))))
    
    c = {}
    for token in tot:
        c.setdefault(token,0)
        c[token] += 1

    n = 50
    tot_words = sum(c.values()) 
    Vocabulary = len(c.keys())

    return tot_words, Vocabulary

def compute_beta_k(v1,v2,n1,n2):
    beta = (log(v1) - log(v2)) / float(log(n1) - log(n2)) 
    try:
        k = float(v1) / n1**beta
    except ZeroDivisionError as e:
        return False # computation error
    return beta, k

def compute_V(beta, k, n):
    return k*n**beta
    
if __name__ == '__main__':
    initStopWords('stopwords') # caution here as to whether this is the right solution!
    
    folder =  GLOBAL['folder']
    folder = folder if folder[-1] == '/' else folder + '/'
    filenames = os.listdir(folder)

    #test(fn=filenames,p=False)

    computed = False
    while (not computed):
        shuffle(filenames)
        sep_line = len(filenames) / 2
        text1 = filenames[:sep_line]
        text2 = filenames[sep_line:]
        
        n1, v1 = summary_output_supressed(text1)
        n2, v2 = summary_output_supressed(text2)
        
        res = compute_beta_k(v1,v2,n1,n2)
        computed = True if res else False

    beta, k = res
    print  "Beta is %.3f, k is %.3f" % (beta, k)
    predict_n = [1000000,100000000]
    for n in predict_n:
        print "The predicted vocabulary size for word size of %10d is %d" % (n, compute_V(beta,k,n))
