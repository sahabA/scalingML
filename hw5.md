
# MIDS - w261 Machine Learning At Scale
__Course Lead:__ Dr James G. Shanahan (__email__ Jimi via  James.Shanahan _AT_ gmail.com)

## Assignment - HW5 Phase1


---
__Name:__  *Sahab Aslam   
__Class:__ MIDS w261 (Section *Your Section Goes Here*, e.g., Fall 2016 Group 1)     
__Email:__  *Your UC Berkeley Email Goes Here*@iSchool.Berkeley.edu     
__Week:__   5

__Due Time:__ 2 Phases. 

* __HW5 Phase 1__ 
This can be done on a local machine (with a unit test on the cloud such as AltaScale's PaaS or on AWS) and is due Tuesday, Week 6 by 8AM (West coast time). It will primarily focus on building a unit/systems and for pairwise similarity calculations pipeline (for stripe documents)

* __HW5 Phase 2__ 
This will require the AltaScale cluster and will be due Tuesday, Week 7 by 8AM (West coast time). 
The focus of  HW5 Phase 2  will be to scale up the unit/systems tests to the Google 5 gram corpus. This will be a group exercise 


# Table of Contents <a name="TOC"></a> 

1.  [HW Intructions](#1)   
2.  [HW References](#2)
3.  [HW Problems](#3)   
1.  [HW Introduction](#1)   
2.  [HW References](#2)
3.  [HW  Problems](#3)   
    1.0.  [HW5.0](#1.0)   
    1.0.  [HW5.1](#1.1)   
    1.2.  [HW5.2](#1.2)   
    1.3.  [HW5.3](#1.3)    
    1.4.  [HW5.4](#1.4)    
    1.5.  [HW5.5](#1.5)    
    1.5.  [HW5.6](#1.6)    
    1.5.  [HW5.7](#1.7)    
    1.5.  [HW5.8](#1.8)    
    1.5.  [HW5.9](#1.9)    
   

<a name="1">
# 1 Instructions
[Back to Table of Contents](#TOC)

MIDS UC Berkeley, Machine Learning at Scale
DATSCIW261 ASSIGNMENT #5

Version 2016-09-25 

 === INSTRUCTIONS for SUBMISSIONS ===
Follow the instructions for submissions carefully.

https://docs.google.com/forms/d/1ZOr9RnIe_A06AcZDB6K1mJN4vrLeSmS2PD6Xm3eOiis/viewform?usp=send_form 


### IMPORTANT

HW4 can be completed locally on your computer

### Documents:
* IPython Notebook, published and viewable online.
* PDF export of IPython Notebook.
    
<a name="2">
# 2 Useful References
[Back to Table of Contents](#TOC)

* See async and live lectures for this week

<a name="3">
# HW Problems
[Back to Table of Contents](#TOC)

## 3.  HW5.0  <a name="1.0"></a>
[Back to Table of Contents](#TOC)

- What is a data warehouse? What is a Star schema? When is it used?

Dataware houses are central repositories for current and historical data
where all data pools from one or more sources and is used for business intelligence to create 
analytical reports.

Star Schema is a fact table which point to the dimension tables (1 to many). It usually is in 3NF. It is used as a basic implementation of an OLAP cube. It can be used to organize the metadata of a relational database.

## 3.  HW5.1  <a name="1.1"></a>
[Back to Table of Contents](#TOC)

- In the database world What is 3NF? Does machine learning use data in 3NF? If so why? 
- In what form does ML consume data?
- Why would one use log files that are denormalized?

A table is in third normal form if:

    A table is in 2nd normal form.
    It contains only columns that are non-transitively dependent on the primary key
Therefore, in 3NF a primiary key is a composite key.

Machine Learning uses 3NF because it saves disk space and it would be time consuming to perform join operations on different tables and then apply the analysis. But ML does not require the data to be in 3NF. Therefore, for performance reasons it is best that log files are denormalized. 

## 3.  HW5.2  <a name="1.2"></a>
[Back to Table of Contents](#TOC)

Using MRJob, implement a hashside join (memory-backed map-side) for left, right and inner joins. Run your code on the  data used in HW 4.4: (Recall HW 4.4: Find the most frequent visitor of each page using mrjob and the output of 4.2  (i.e., transfromed log file). In this output please include the webpage URL, webpageID and Visitor ID.)

Justify which table you chose as the Left table in this hashside join.

Please report the number of rows resulting from:

- (1) Left joining Table Left with Table Right
- (2) Right joining Table Left with Table Right
- (3) Inner joining Table Left with Table Right


```python
%%writefile transactions.dat
Alice Bob|$10|US
Sam Sneed|$1|CA
Jon Sneed|$20|CA
Arnold Wesise|$400|UK
Henry Bob|$2|US
Yo Yo Ma|$2|CA
Jon York|$44|CA
Alex Ball|$5|UK
Jim Davis|$66|JA
```

    Overwriting transactions.dat



```python
%%writefile Countries.dat
United States|US
Canada|CA
United Kingdom|UK
Italy|IT
```

    Overwriting Countries.dat



```python
%%writefile lj52.py
from mrjob.job import MRJob
 
class leftjoin2(MRJob):
    def mapper(self, _, line):
        x = line.split("|")
        if len(x) == 3:
            yield x[2], ("lefttable", x[0], x[1])
        else:
            yield x[1], ("righttable",  x[0] )

    def reducer(self, key, values):
        countries = list()
        orders = list()
        for val in values:
            if val[0]== u'lefttable':
                countries.append(val)
            else:
                orders.append(val)

        for c in countries:
            if len(orders)==0:
                yield None, [key] + c[1:] + [None]
            else:
                for o in orders:
                    yield None, [key] + c[1:] + o[1:]
if __name__ == '__main__':
    leftjoin2.run()
```

    Overwriting lj52.py



```python
%load_ext autoreload
%autoreload 2
from lj52 import leftjoin2
mr_job = leftjoin2(args=['Countries.dat','transactions.dat'])
with mr_job.make_runner() as runner: 
    runner.run()
    count = 0
    # stream_output: get access of the output 
    for line in runner.stream_output():
        key,value =  mr_job.parse_output_line(line)
        print value
        count = count + 1
print "\n"
print "There are %s records" %count
```

    ['CA', 'Jon Sneed', '$20', 'Canada']
    ['CA', 'Jon York', '$44', 'Canada']
    ['CA', 'Sam Sneed', '$1', 'Canada']
    ['CA', 'Yo Yo Ma', '$2', 'Canada']
    ['JA', 'Jim Davis', '$66', None]
    ['UK', 'Alex Ball', '$5', 'United Kingdom']
    ['UK', 'Arnold Wesise', '$400', 'United Kingdom']
    ['US', 'Alice Bob', '$10', 'United States']
    ['US', 'Henry Bob', '$2', 'United States']
    
    
    There are 9 records



```python
%%writefile rj52.py
from mrjob.job import MRJob
 
class rightjoin(MRJob):
    def mapper(self, _, line):
        x = line.split("|")
        if len(x) == 3:
            yield x[2], ("lefttable", x[0], x[1])
        else:
            yield x[1], ("righttable",  x[0] )

    def reducer(self, key, values):
        countries = list()
        orders = list()
        for val in values:
            if val[0]== u'lefttable':
                countries.append(val)
            else:
                orders.append(val)

        
        for o in orders:
            if len(countries)==0:
                yield None, [key] + o[1:] + [None]
            else:
                for c in countries:
                    yield None, [key] + c[1:] + o[1:]
if __name__ == '__main__':
    rightjoin.run()
```

    Overwriting rj52.py



```python
from rj52 import rightjoin
mr_job = rightjoin(args=['Countries.dat','transactions.dat'])
with mr_job.make_runner() as runner: 
    runner.run()
    count = 0
    # stream_output: get access of the output 
    for line in runner.stream_output():
        key,value =  mr_job.parse_output_line(line)
        print value
        count = count + 1
print "\n"
print "There are %s records" %count
```

    ['CA', 'Jon Sneed', '$20', 'Canada']
    ['CA', 'Jon York', '$44', 'Canada']
    ['CA', 'Sam Sneed', '$1', 'Canada']
    ['CA', 'Yo Yo Ma', '$2', 'Canada']
    ['IT', 'Italy', None]
    ['UK', 'Alex Ball', '$5', 'United Kingdom']
    ['UK', 'Arnold Wesise', '$400', 'United Kingdom']
    ['US', 'Alice Bob', '$10', 'United States']
    ['US', 'Henry Bob', '$2', 'United States']
    
    
    There are 9 records



```python
%%writefile ij52.py
from mrjob.job import MRJob
 
class innerjoin(MRJob):
    def mapper(self, _, line):
        x = line.split("|")
        if len(x) == 3:
            yield x[2], ("lefttable", x[0], x[1])
        else:
            yield x[1], ("righttable",  x[0] )

    def reducer(self, key, values):
        countries = list()
        orders = list()
        for val in values:
            if val[0]== u'lefttable':
                countries.append(val)
            else:
                orders.append(val)

        for c in countries:
            for o in orders:
                yield None, [key] + o[1:] + c[1:]
if __name__ == '__main__':
    innerjoin.run()
```

    Overwriting ij52.py



```python
%load_ext autoreload
%autoreload 2
from ij52 import innerjoin
mr_job = innerjoin(args=['Countries.dat','transactions.dat'])
with mr_job.make_runner() as runner: 
    runner.run()
    count = 0
    # stream_output: get access of the output 
    for line in runner.stream_output():
        key,value =  mr_job.parse_output_line(line)
        print value
        count = count + 1
print "\n"
print "There are %s records" %count
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload
    ['CA', 'Canada', 'Jon Sneed', '$20']
    ['CA', 'Canada', 'Jon York', '$44']
    ['CA', 'Canada', 'Sam Sneed', '$1']
    ['CA', 'Canada', 'Yo Yo Ma', '$2']
    ['UK', 'United Kingdom', 'Alex Ball', '$5']
    ['UK', 'United Kingdom', 'Arnold Wesise', '$400']
    ['US', 'United States', 'Alice Bob', '$10']
    ['US', 'United States', 'Henry Bob', '$2']
    
    
    There are 8 records


## 3.  HW5.3 <a name="1.3"></a> Systems tests on n-grams dataset (Phase1) and full experiment (Phase 2)
[Back to Table of Contents](#TOC)

## 3.  HW5.3.0 Run Systems tests locally (PHASE1)
[Back to Table of Contents](#TOC)

A large subset of the Google n-grams dataset

https://aws.amazon.com/datasets/google-books-ngrams/

which we have placed in a bucket/folder on Dropbox and on s3:

https://www.dropbox.com/sh/tmqpc4o0xswhkvz/AACUifrl6wrMrlK6a3X3lZ9Ea?dl=0 

s3://filtered-5grams/

In particular, this bucket contains (~200) files (10Meg each) in the format:

	(ngram) \t (count) \t (pages_count) \t (books_count)

The next cell shows the first 10 lines of the googlebooks-eng-all-5gram-20090715-0-filtered.txt file.


__DISCLAIMER__: Each record is already a 5-gram. We should calculate the stripes cooccurrence data from the raw text and not from the 5-gram preprocessed data. Calculatating pairs on this 5-gram is a little corrupt as we will be double counting cooccurences. Having said that this exercise can still pull out some simialr terms. 


```python

```

#### 1: unit/systems first-10-lines


```python
%writefile googlebooks-eng-all-5gram-20090715-0-filtered-first-10-lines.txt
A BILL FOR ESTABLISHING RELIGIOUS	59	59	54
A Biography of General George	92	90	74
A Case Study in Government	102	102	78
A Case Study of Female	447	447	327
A Case Study of Limited	55	55	43
A Child's Christmas in Wales	1099	1061	866
A Circumstantial Narrative of the	62	62	50
A City by the Sea	62	60	49
A Collection of Fairy Tales	123	117	80
A Collection of Forms of	116	103	82
```


      File "<ipython-input-1-0d227eed86ca>", line 2
        A BILL FOR ESTABLISHING RELIGIOUS	59	59	54
             ^
    SyntaxError: invalid syntax



For _HW 5.4-5.5_,  unit test and regression test your code using the  followings small test datasets:

* googlebooks-eng-all-5gram-20090715-0-filtered.txt [see above]
* stripe-docs-test [see below]
* atlas-boon-test [see below]

#### 2: unit/systems atlas-boon


```python
%%writefile atlas-boon-systems-test.txt
atlas boon	50	50	50
boon cava dipped	10	10	10
atlas dipped	15	15	15
```

    Overwriting atlas-boon-systems-test.txt


#### 3: unit/systems stripe-docs-test
Three terms, A,B,C and their corresponding stripe-docs of co-occurring terms

- DocA {X:20, Y:30, Z:5}
- DocB {X:100, Y:20}
- DocC {M:5, N:20, Z:5}


```python
############################################
# Stripes for systems test 1 (predefined)
############################################

with open("mini_stripes.txt", "w") as f:
    f.writelines([
        '"DocA"\t{"X":20, "Y":30, "Z":5}\n',
        '"DocB"\t{"X":100, "Y":20}\n',  
        '"DocC"\t{"M":5, "N":20, "Z":5, "Y":1}\n'
    ])
!cat mini_stripes.txt   
```

    "DocA"	{"X":20, "Y":30, "Z":5}
    "DocB"	{"X":100, "Y":20}
    "DocC"	{"M":5, "N":20, "Z":5, "Y":1}



```python
%%writefile buildStripes.py

#import csv, sys
from mrjob.job import MRJob
#from mrjob.step import MRStep
#from mrjob.compat import get_jobconf_value

class buildStripes(MRJob):
          
    #yields words = "apple blu pink"
    # to apple {'blu': 90} apple {'pink': 90}  blu {'apple': 90}
    def mapper(self, _, lines):
        words, count, pages, books = lines.split("\t")
        count = int(count)
        pages = int(pages)
        books = int(books)
        allwords = words.split()
        for i, word in enumerate(allwords):
            other_words = allwords[:]
            other_words.pop(i)
            x = {}
            for each_word in other_words:
                y={}
                y[each_word] = count
                x = {m: x.get(m, 0) + y.get(m,0) for m in set(x) | set(y)}
            yield word, x
                    
    
    def reducer(self, keys, values):
        stripes = {}
        x = {}
        for y in values:
            x = {m: x.get(m, 0) + y.get(m,0) for m in set(x) | set(y)}
        print keys, '\t', x
        
if __name__ == "__main__":
    buildStripes.run()
        
 
```

    Overwriting buildStripes.py



```python
!cat atlas-boon-systems-test.txt | python buildStripes.py -q > atlas_stripes.txt
!cat atlas_stripes.txt
```

    atlas 	{'dipped': 15, 'boon': 50}
    boon 	{'atlas': 50, 'dipped': 10, 'cava': 10}
    cava 	{'dipped': 10, 'boon': 10}
    dipped 	{'atlas': 15, 'boon': 10, 'cava': 10}



```python
###############################################
# Make Stripes from ngrams for systems test 2
###############################################

!aws s3 rm --recursive s3://ucb261-hw5/hw5-4-stripes-mj
!python buildStripes.py -r emr mj_systems_test.txt \
    --cluster-id=j-2WHMJSLZDGOY5 \
    --output-dir=s3://ucb261-hw5/hw5-4-stripes-mj \
    --file=stopwords.txt \
    --file=mostFrequent/part-00000 \
# Output suppressed    
```

    /bin/sh: aws: command not found
    python: can't open file 'buildStripes.py': [Errno 2] No such file or directory


## TASK: Phase 1
Complete 5.4 and 5.5 and systems test them using the above test datasets. Phase 2 will focus on the entire Ngram dataset.

To help you through these tasks please verify that your code gives the following results (for stripes, inverted index, and pairwise similarities).

#### Step 10  Build an cooccureence strips from the atlas-boon


```python
#Using the atlas-boon systems test
atlas boon	50	50	50
boon cava dipped	10	10	10
atlas dipped	15	15	15
```

#### Stripe documents for  atlas-boon systems test


```python
###############################################
# Make Stripes from ngrams 
###############################################
!aws s3 rm --recursive s3://ucb261-hw5/hw5-4-stripes-mj
!python buildStripes.py -r emr mj_systems_test.txt \
    --cluster-id=j-2WHMJSLZDG \
    --output-dir=s3://ucb261-hw5/hw5-4-stripes-mj \
    --file=stopwords.txt \
    --file=mostFrequent/part-00000 \
# Output suppressed    
```


```python
!mkdir stripes-mj
!aws s3 sync s3://ucb261-hw5/hw5-4-stripes-mj/  stripes-mj/
!cat stripes-mj/part-*
```


```python
"atlas"	{"dipped": 15, "boon": 50}
"boon"	{"atlas": 50, "dipped": 10, "cava": 10}
"cava"	{"dipped": 10, "boon": 10}
"dipped"	{"atlas": 15, "boon": 10, "cava": 10}
```

## Building stripes execution MR stats: (report times!)
    took ~11 minutes on 5 m3.xlarge nodes
    Data-local map tasks=188
	Launched map tasks=190
	Launched reduce tasks=15
	Other local map tasks=2

#### Step 20  create inverted index, and calculate pairwise similarity


```python
%%writefile InvertIndex.py

from mrjob.job import MRJob
from mrjob.protocol import JSONProtocol
from collections import Counter

class InvertIndex(MRJob):
    #MRJob.input_protocol = JSONProtocol
    
    def mapper(self, key, lines):
        key, words = lines.split("\t")
        words = words.strip('{').strip('}')
        words= words.split(',')
        corpus_length = len(words)
        for word in words: 
           # print 
            yield word.split(":")[0].strip().strip("'"), {key.strip('"').strip('"'):corpus_length}
            

    def reducer(self, keys, values):
        stripes = {}
        x = {}
        for y in values:
            x = {m: x.get(m, 0) + y.get(m,0) for m in set(x) | set(y)}
        print keys, '\t', x
        
if __name__ == "__main__":
    InvertIndex.run()


```

    Overwriting InvertIndex.py



```python
!cat atlas_stripes.txt | python InvertIndex.py -q > atlas_inverted.txt
!cat atlas_inverted.txt


```

    atlas 	{'dipped ': 3, 'boon ': 3}
    boon 	{'atlas ': 2, 'cava ': 2, 'dipped ': 3}
    cava 	{'dipped ': 3, 'boon ': 3}
    dipped 	{'atlas ': 2, 'boon ': 3, 'cava ': 2}


<p><strong>Solution 1:</strong> </p>
<ol>
<li>Create an Inverted Index. </li>
<li>Use the output to calculate similarities. </li>
<li>Build custom partitioner, re-run the similarity code, and output total order sorted partitions.</li>
</ol>


### Inverted Index

### Pairwise Similairity 


```python
%%writefile Similarity.py

from mrjob.job import MRJob
class Similarity(MRJob):
    
    doc = list()
    
    def mapper(self, _, lines):
        key, words = lines.split("\t")
        words = words.strip('{').strip('}')
        #print words
        words= words.split(',')
        items = []
        for word in words: 
            word = word.strip().strip("'").strip('"')
            
            items.append(word)
        if(len(items) >1):
            k = [(items[i],items[j] ) for i in range(len(items)) for j in range(i+1, len(items))]
            for m in k:
                 yield m, 1

        
    def combiner(self, key, values):
        count = 0
        for v in values:
            count += v
        yield key, count
        
    def reducer(self, key, values):
        pairs = list()
        total = 0
        aplusb = 0
        aplusb_cosine = 1
        overlap = 0
        average = 0
        for v in values:
            total = total + v
        #counts = {}
        for d in key:
            doc, count = d.split(":")
            count = int(count.strip("'").strip())
            aplusb = aplusb + count
            aplusb_cosine = aplusb_cosine * (count**.5)
            if(overlap == 0 or overlap > count ):
                overlap = count
                #print overlap
            pairs.append(doc)
        calc={}
        j = total*1.0/(aplusb-total)
        c = total*1.0/(aplusb_cosine)
        d = 2.0*total/aplusb
        o = 1.0*total/overlap
        average = (j + c + d + o)/4
        calc["average"] = average
        calc["jaccard"] = j
        calc["cosine"] = c
        calc["dice"] = d
        calc["overlap"] = o
        
        
        print pairs,"\t", calc
        #print "******"*5
        #print "done"
        
            

        
if __name__ == "__main__":
    Similarity.run()


```

    Overwriting Similarity.py



```python
#!cat mini_stripes.txt | python InvertIndex.py -q > mini_stripes_inverted.txt
#!cat mini_stripes_inverted.txt
!cat mini_stripes_inverted.txt | python Similarity.py -q --jobconf mapred.reduce.tasks=1 > miniS1.txt
```


```python
!cat miniS1.txt
```

    ["DocB'", "DocA'"] 	{'average': 0.8207908118985981, 'cosine': 0.8164965809277259, 'dice': 0.8, 'overlap': 1.0, 'jaccard': 0.6666666666666666}
    ["DocB'", "DocC'"] 	{'average': 0.34672168098165174, 'cosine': 0.35355339059327373, 'dice': 0.3333333333333333, 'overlap': 0.5, 'jaccard': 0.2}
    ["DocC'", "DocA'"] 	{'average': 0.553861376821216, 'cosine': 0.5773502691896258, 'dice': 0.5714285714285714, 'overlap': 0.6666666666666666, 'jaccard': 0.4}



```python
!cat atlas_inverted.txt | python Similarity.py -q --jobconf mapred.reduce.tasks=1
```

    ["atlas '", "boon '"] {'average': 0.38956207261596576, 'cosine': 0.40824829046386296, 'dice': 0.4, 'overlap': 0.5, 'jaccard': 0.25}
    ["atlas '", "cava '"] {'average': 1.0, 'cosine': 0.9999999999999998, 'dice': 1.0, 'overlap': 1.0, 'jaccard': 1.0}
    ["atlas '", "dipped '"] {'average': 0.38956207261596576, 'cosine': 0.40824829046386296, 'dice': 0.4, 'overlap': 0.5, 'jaccard': 0.25}
    ["boon '", "cava '"] {'average': 0.38956207261596576, 'cosine': 0.40824829046386296, 'dice': 0.4, 'overlap': 0.5, 'jaccard': 0.25}
    ["cava '", "dipped '"] {'average': 0.38956207261596576, 'cosine': 0.40824829046386296, 'dice': 0.4, 'overlap': 0.5, 'jaccard': 0.25}
    ["dipped '", "boon '"] {'average': 0.625, 'cosine': 0.6666666666666667, 'dice': 0.6666666666666666, 'overlap': 0.6666666666666666, 'jaccard': 0.5}


## 3.  HW5.3.1  <a name="1.3"></a> Run systems tests on the CLOUD  (PHASE 1)
[Back to Table of Contents](#TOC)

Repeat HW5.3.0 on the cloud (AltaScale / AWS/ SoftLayer/ Azure). Make sure all tests give correct results

# PHASE 2: Full-scale experiment on Google N-gram data

__ Once you are happy with your test results __ proceed to generating  your results on the Google n-grams dataset. 

## 3.  HW5.3.2  Full-scale experiment: EDA of Google n-grams dataset (PHASE 2)
[Back to Table of Contents](#TOC)

Do some EDA on this dataset using mrjob, e.g., 

- Longest 5-gram (number of characters)
- Top 10 most frequent words (please use the count information), i.e., unigrams
- 20 Most/Least densely appearing words (count/pages_count) sorted in decreasing order of relative frequency 
- Distribution of 5-gram sizes (character length).  E.g., count (using the count field) up how many times a 5-gram of 50 characters shows up. Plot the data graphically using a histogram.




```python

```

## 3.  HW5.3.4 OPTIONAL Question: log-log plots (PHASE 2)
[Back to Table of Contents](#TOC)

Plot the log-log plot of the frequency distributuion of unigrams. Does it follow power law distribution?

For more background see:
- https://en.wikipedia.org/wiki/Log%E2%80%93log_plot
- https://en.wikipedia.org/wiki/Power_law

## 3.  HW5.4  <a name="1.4"></a> Synonym detection over 2Gig of Data
[Back to Table of Contents](#TOC)

For the remainder of this assignment please feel free to eliminate stop words from your analysis

>There is also a corpus of stopwords, that is, high-frequency words like "the", "to" and "also" that we sometimes want to filter out of a document before further processing. Stopwords usually have little lexical content, and their presence in a text fails to distinguish it from other texts. Python's nltk comes with a prebuilt list of stopwords (see below). Using this stopword list filter out these tokens from your analysis and rerun the experiments in 5.5 and disucuss the results of using a stopword list and without using a stopword list.

> from nltk.corpus import stopwords
 stopwords.words('english')
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

### 2: A large subset of the Google n-grams dataset as was described above

For each HW 5.4 -5.5.1 Please unit test and system test your code with respect 
to SYSTEMS TEST DATASET and show the results. 
Please compute the expected answer by hand and show your hand calculations for the 
SYSTEMS TEST DATASET. Then show the results you get with your system.

In this part of the assignment we will focus on developing methods for detecting synonyms, using the Google 5-grams dataset. At a high level:


1. remove stopwords
2. get 10,0000 most frequent
3. get 1000 (9001-10000) features
3. build stripes

To accomplish this you must script two main tasks using MRJob:


__TASK (1)__ Build stripes for the most frequent 10,000 words using cooccurence information based on
the words ranked from 9001,-10,000 as a basis/vocabulary (drop stopword-like terms),
and output to a file in your bucket on s3 (bigram analysis, though the words are non-contiguous).


__TASK (2)__ Using two (symmetric) comparison methods of your choice 
(e.g., correlations, distances, similarities), pairwise compare 
all stripes (vectors), and output to a file in your bucket on s3.

#### Design notes for TASK (1)
For this task you will be able to modify the pattern we used in HW 3.2
(feel free to use the solution as reference). To total the word counts 
across the 5-grams, output the support from the mappers using the total 
order inversion pattern:

<*word,count>

to ensure that the support arrives before the cooccurrences.

In addition to ensuring the determination of the total word counts,
the mapper must also output co-occurrence counts for the pairs of
words inside of each 5-gram. Treat these words as a basket,
as we have in HW 3, but count all stripes or pairs in both orders,
i.e., count both orderings: (word1,word2), and (word2,word1), to preserve
symmetry in our output for TASK (2).

#### Design notes for _TASK (2)_
For this task you will have to determine a method of comparison.
Here are a few that you might consider:

- Jaccard
- Cosine similarity
- Spearman correlation
- Euclidean distance
- Taxicab (Manhattan) distance
- Shortest path graph distance (a graph, because our data is symmetric!)
- Pearson correlation
- Kendall correlation

However, be cautioned that some comparison methods are more difficult to
parallelize than others, and do not perform more associations than is necessary, 
since your choice of association will be symmetric.

Please use the inverted index (discussed in live session #5) based pattern to compute the pairwise (term-by-term) similarity matrix. 

Please report the size of the cluster used and the amount of time it takes to run for the index construction task and for the synonym calculation task. How many pairs need to be processed (HINT: use the posting list length to calculate directly)? Report your  Cluster configuration!

## 3.  HW5.5  <a name="1.5"></a> Evaluation of synonyms that your discovered
[Back to Table of Contents](#TOC)


In this part of the assignment you will evaluate the success of you synonym detector (developed in response to HW5.4).
Take the top 1,000 closest/most similar/correlative pairs of words as determined by your measure in HW5.4, and use the synonyms function in the accompanying python code:

nltk_synonyms.py

Note: This will require installing the python nltk package:

http://www.nltk.org/install.html

and downloading its data with nltk.download().

For each (word1,word2) pair, check to see if word1 is in the list, 
synonyms(word2), and vice-versa. If one of the two is a synonym of the other, 
then consider this pair a 'hit', and then report the precision, recall, and F1 measure  of 
your detector across your 1,000 best guesses. Report the macro averages of these measures.

### Calculate performance measures:
$$Precision (P) = \frac{TP}{TP + FP} $$  
$$Recall (R) = \frac{TP}{TP + FN} $$  
$$F1 = \frac{2 * ( precision * recall )}{precision + recall}$$


We calculate Precision by counting the number of hits and dividing by the number of occurances in our top1000 (opportunities)   
We calculate Recall by counting the number of hits, and dividing by the number of synonyms in wordnet (syns)


Other diagnostic measures not implemented here:  https://en.wikipedia.org/wiki/F1_score#Diagnostic_Testing


```python
''' Performance measures '''
from __future__ import division
import numpy as np
import json
import nltk
from nltk.corpus import wordnet as wn
import sys
#print all the synset element of an element
def synonyms(string):
    syndict = {}
    for i,j in enumerate(wn.synsets(string)):
        syns = j.lemma_names()
        for syn in syns:
            syndict.setdefault(syn,1)
    return syndict.keys()
hits = []

TP = 0
FP = 0

TOTAL = 0
flag = False # so we don't double count, but at the same time don't miss hits

## For this part we can use one of three outputs. They are all the same, but were generated differently
# 1. the top 1000 from the full sorted dataset -> sortedSims[:1000]
# 2. the top 1000 from the partial sort aggragate file -> sims2/top1000sims
# 3. the top 1000 from the total order sort file -> head -1000 sims_parts/part-00004

top1000sims = []
with open("sims2/top1000sims","r") as f:
    for line in f.readlines():

        line = line.strip()
        avg,lisst = line.split("\t")
        lisst = json.loads(lisst)
        lisst.append(avg)
        top1000sims.append(lisst)
    

measures = {}
not_in_wordnet = []

for line in top1000sims:
    TOTAL += 1

    pair = line[0]
    words = pair.split(" - ")
    
    for word in words:
        if word not in measures:
            measures[word] = {"syns":0,"opps": 0,"hits":0}
        measures[word]["opps"] += 1 
    
    syns0 = synonyms(words[0])
    measures[words[1]]["syns"] = len(syns0)
    if len(syns0) == 0:
        not_in_wordnet.append(words[0])
        
    if words[1] in syns0:
        TP += 1
        hits.append(line)
        flag = True
        measures[words[1]]["hits"] += 1
        
        
        
    syns1 = synonyms(words[1]) 
    measures[words[0]]["syns"] = len(syns1)
    if len(syns1) == 0:
        not_in_wordnet.append(words[1])

    if words[0] in syns1:
        if flag == False:
            TP += 1
            hits.append(line)
            measures[words[0]]["hits"] += 1
            
    flag = False    

precision = []
recall = []
f1 = []

for key in measures:
    p,r,f = 0,0,0
    if measures[key]["hits"] > 0 and measures[key]["syns"] > 0:
        p = measures[key]["hits"]/measures[key]["opps"]
        r = measures[key]["hits"]/measures[key]["syns"]
        f = 2 * (p*r)/(p+r)
    
    # For calculating measures, only take into account words that have synonyms in wordnet
    if measures[key]["syns"] > 0:
        precision.append(p)
        recall.append(r)
        f1.append(f)

    
# Take the mean of each measure    
print "—"*110    
print "Number of Hits:",TP, "out of top",TOTAL
print "Number of words without synonyms:",len(not_in_wordnet)
print "—"*110 
print "Precision\t", np.mean(precision)
print "Recall\t\t", np.mean(recall)
print "F1\t\t", np.mean(f1)
print "—"*110  

print "Words without synonyms:"
print "-"*100

for word in not_in_wordnet:
    print synonyms(word),word

    
```

### Sample output


```python
——————————————————————————————————————————————————————————————————————————————————————————————————————————————
Number of Hits: 31 out of top 1000
Number of words without synonyms: 67
——————————————————————————————————————————————————————————————————————————————————————————————————————————————
Precision	0.0280214404967
Recall		0.0178598869579
F1		0.013965517619
——————————————————————————————————————————————————————————————————————————————————————————————————————————————
Words without synonyms:
----------------------------------------------------------------------------------------------------
[] scotia
[] hong
[] kong
[] angeles
[] los
[] nor
[] themselves
[] 
.......
```

## 3.  HW5.6  <a name="1.6"></a> OPTIONAL: using different vocabulary subsets
[Back to Table of Contents](#TOC)


Repeat HW5 using vocabulary words ranked from 8001,-10,000;  7001,-10,000; 6001,-10,000; 5001,-10,000; 3001,-10,000; and 1001,-10,000;
Dont forget to report you Cluster configuration.

Generate the following graphs:
-- vocabulary size (X-Axis) versus CPU time for indexing
-- vocabulary size (X-Axis) versus number of pairs processed
-- vocabulary size (X-Axis) versus F1 measure, Precision, Recall


```python

```

## 3.  HW5.7  <a name="1.7"></a> OPTIONAL: filter stopwords
[Back to Table of Contents](#TOC)

There is also a corpus of stopwords, that is, high-frequency words like "the", "to" and "also" that we sometimes want to filter out of a document before further processing. Stopwords usually have little lexical content, and their presence in a text fails to distinguish it from other texts. Python's nltk comes with a prebuilt list of stopwords (see below). Using this stopword list filter out these tokens from your analysis and rerun the experiments in 5.5 and disucuss the results of using a stopword list and without using a stopword list.

> from nltk.corpus import stopwords
>> stopwords.words('english')
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']




```python

```



## 3.  HW5.8 <a name="1.8"></a> OPTIONAL 
[Back to Table of Contents](#TOC)

There are many good ways to build our synonym detectors, so for this optional homework, 
measure co-occurrence by (left/right/all) consecutive words only, 
or make stripes according to word co-occurrences with the accompanying 
2-, 3-, or 4-grams (note here that your output will no longer 
be interpretable as a network) inside of the 5-grams.


```python

```

## 3.  HW5.9 <a name="1.9"></a> OPTIONAL 
[Back to Table of Contents](#TOC)

Once again, benchmark your top 10,000 associations (as in 5.5), this time for your
results from 5.6. Has your detector improved?


```python

```
