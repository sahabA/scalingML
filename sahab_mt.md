
# MIDS Machine Learning at Scale
## MidTerm Exam  

12:15PM - 2:00PM(psT)
October 19, 2016   
Midterm

MIDS Machine Learning at Scale



### Please insert your contact information here
Name: Sahab Aslam   
email: sahabaslam@hotmail.com 
UC Berkeley ID: 26061144


```python
import numpy as np
from __future__ import division

%reload_ext autoreload
%autoreload 2

```

# Exam Instructions

1. : Please insert Name and Email address in the first cell of this notebook
2. : Please acknowledge receipt of exam by sending a quick email reply to the instructor
3. : Review the submission form first to scope it out (it will take a 5-10 minutes to input your 
   answers and other information into this form): 

    + [Exam Submission form](http://goo.gl/forms/ggNYfRXz0t) 

4. : Please keep all your work and responses in ONE (1) notebook only (and submit via the submission form)
5. : Please make sure that the NBViewer link for your Submission notebook works 
6. : Please submit your solutions and notebook via the following form:

     + [Exam Submission form](http://goo.gl/forms/ggNYfRXz0t)

7. : For the midterm you will need access to MrJob and Jupyter on your local machines or on AltaScale/AWS to complete some of the questions (like fill in the code to do X).
8. : As for question types:
    + Knowledge test Programmatic/doodle (take photos; embed the photos in your notebook) 
    + All programmatic questions can be run locally on your laptop (using MrJob only) or on the cluster

9. : This is an open book exam meaning you can consult webpages and textbooks, class notes, slides etc. but you can not discuss with each other or any other person/group. If any collusion, then this will result in a zero grade and will be grounds for dismissal from the entire program. Please complete this exam by yourself within the time limit. 

# Exam questions begins here

===Map-Reduce===

### MT1. Which of the following statememts about map-reduce are true? 

(I) If you only have 1 computer with 1 computing core, then map-reduce is unlikely to help   
(II) If we run map-reduce using N single-core computers, then it is likely to get at least an N-Fold speedup compared to using 1 computer   
(III) Because of network latency and other overhead associated with map-reduce, if we run map-reduce using N    computers, then we will get less than N-Fold speedup compared to using 1 computer   
(IV) When using map-reduce for learning a naive Bayes classifier for SPAM classification, we usually use a single machine that accumulates the partial class and word stats from each of the map machines, in order to compute the final model.

Please select one from the following that is most correct:

* (a) I, II, III, IV
* (b) I, III, IV
* (c) I, III
* (d) I,II, III

#####  MT1. a  #  ####
<span style="font-weight:bold; color:green;">C

NOTE that it is Not necessary to provide an answer in this cell. But as you can imagine putting in some notes enables you to cross check your logic later.
</SPAN>

===Order inversion===

### MT2. normalized product co-occurrence 

uppose you wish to write a MapReduce job that creates  normalized product co-occurrence (i.e., pairs of products that have been purchased together) data form a large transaction file of shopping baskets. In addition, we want the relative frequency of coocurring products. Given this scenario, to ensure that all (potentially many) reducers
receive appropriate normalization factors (denominators)for a product
in the most effcient order in their input streams (so as to minimize memory overhead on the reducer side), 
the mapper should emit/yield records according to which pattern for the product occurence totals:   

(a) emit (\*,product) count   
(b) There is no need to use  order inversion here   
(c) emit (product,\*) count   
(d) None of the above   

A

===Map-Reduce===

### MT3. What is the input to the Reduce function in MRJob? Select the most correct choice.  

   
(a) An arbitrarily sized list of key/value pairs.    
(b) One key and a list of some values associated with that key  
(c) One key and a list of all values associated with that key.   
(d) None of the above   

C

===Bayesian document classification===   
  
### MT4. When building a Bayesian document classifier, Laplace smoothing serves what purpose?   

(a) It allows you to use your training data as your validation data.   
(b) It prevents zero-products in the posterior distribution.  
(c) It accounts for words that were missed by regular expressions.    
(d) None of the above   

#B

### MT5. Big Data
Big data is defined as the voluminous amount of structured, unstructured or semi-structured data that has huge potential for mining but is so large that it cannot be processed nor stored using traditional (single computer) computing and storage systems. Big data is characterized by its high velocity, volume and variety that requires cost effective and innovative methods for information processing to draw meaningful business insights. More than the volume of the data – it is the nature of the data that defines whether it is considered as Big Data or not. What do the four V’s of Big Data denote? Here is a potential simple explanation for each of the four critical features of big data (some or all of which is correct):

__Statements__ 
* (I)  Volume –Scale of data
* (II) Velocity – Batch processing of data offline
* (III)Variety – Different forms of data
* (IV) Veracity –Uncertainty of data

Which combination of the above statements is correct. Select a single correct response from the following :

* (a) I, II, III, IV
* (b) I, III, IV
* (c) I, III
* (d) I,II, III

A)  

### MT6. Combiners can be integral to the successful utilization of the Hadoop shuffle.  
Using combiners result in what?  

* (I) minimization of reducer workload     
* (II) minimization of disk storage for mapper results   
* (III) minimization of network traffic    
* (IV) none of the above 

Select most correct option (i.e., select one option only) from the following:

* (a) I   
* (b) I, II and III   
* (c) II and III   
* (d) IV   


B) II => III => I

## Pairwise similarity using K-L divergence

In probability theory and information theory, the Kullback–Leibler divergence 
(also information divergence, information gain, relative entropy, KLIC, or KL divergence) 
is a non-symmetric measure of the difference between two probability distributions P and Q. 
Specifically, the Kullback–Leibler divergence of Q from P, denoted DKL(P\‖Q), 
is a measure of the information lost when Q is used to approximate P:

For discrete probability distributions P and Q, 
the Kullback–Leibler divergence of Q from P is defined to be

    + KLDistance(P, Q) = Sum_over_item_i (P(i) log (P(i) / Q(i))      

In the extreme cases, the KL Divergence is 1 when P and Q are maximally different
and is 0 when the two distributions are exactly the same (follow the same distribution).

For more information on K-L Divergence see:

    + [K-L Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)

For the next three question we will use an MRjob class for calculating pairwise similarity 
using K-L Divergence as the similarity measure:

* Job 1: create inverted index (assume just two objects)
* Job 2: calculate/accumulate the similarity of each pair of objects using K-L Divergence


Using the following cells  then fill in the code for the first reducer to calculate 
the K-L divergence of objects (letter documents) in line1 and line2, i.e., KLD(Line1||line2).

Here we ignore characters which are not alphabetical. And all alphabetical characters are lower-cased in the first mapper.

## Using the MRJob Class below  calculate the  KL divergence of the following two string objects.


```python
%%writefile kltext.txt
1.Data Science is an interdisciplinary field about processes and systems to extract knowledge or insights from large volumes of data in various forms (data in various forms, data in various forms, data in various forms), either structured or unstructured,[1][2] which is a continuation of some of the data analysis fields such as statistics, data mining and predictive analytics, as well as Knowledge Discovery in Databases.
2.Machine learning is a subfield of computer science[1] that evolved from the study of pattern recognition and computational learning theory in artificial intelligence.[1] Machine learning explores the study and construction of algorithms that can learn from and make predictions on data.[2] Such algorithms operate by building a model from example inputs in order to make data-driven predictions or decisions,[3]:2 rather than following strictly static program instructions.
```

    Writing kltext.txt


## MRjob class for calculating pairwise similarity using K-L Divergence as the similarity measure

Job 1: create inverted index (assume just two objects) <P>
Job 2: calculate the similarity of each pair of objects 


```python
import numpy as np
np.log(3)
```




    1.0986122886681098




```python
!cat kltext.txt
```

    1.Data Science is an interdisciplinary field about processes and systems to extract knowledge or insights from large volumes of data in various forms (data in various forms, data in various forms, data in various forms), either structured or unstructured,[1][2] which is a continuation of some of the data analysis fields such as statistics, data mining and predictive analytics, as well as Knowledge Discovery in Databases.
    2.Machine learning is a subfield of computer science[1] that evolved from the study of pattern recognition and computational learning theory in artificial intelligence.[1] Machine learning explores the study and construction of algorithms that can learn from and make predictions on data.[2] Such algorithms operate by building a model from example inputs in order to make data-driven predictions or decisions,[3]:2 rather than following strictly static program instructions.


```python
%%writefile kldivergence.py
from __future__ import division
from mrjob.job import MRJob
import re
import numpy as np
class kldivergence(MRJob):
    
    # process each string character by character
    # the relative frequency of each character emitting Pr(character|str)
    # for input record 1.abcbe
    # emit "a"    [1, 0.2]
    # emit "b"    [1, 0.4] etc...
    def mapper1(self, _, line):
        index = int(line.split('.',1)[0])
        letter_list = re.sub(r"[^A-Za-z]+", '', line).lower()
        count = {}
        for l in letter_list:
            if count.has_key(l):
                count[l] += 1
            else:
                count[l] = 1
        for key in count:
            yield key, [index, count[key]*1.0/len(letter_list)]


    def reducer1(self, key, values):
        p = 0
        q = 0
        for v in values:
            if v[0] == 1:  #String 1
                p = v[1]
            else:          # String 2
                q = v[1]
                
        yield None, p*np.log(p/q)

    #Aggegate components            
    def reducer2(self, key, values):
        kl_sum = 0
        for value in values:
            kl_sum = kl_sum + value
        yield "KLDivergence", kl_sum
            
    def steps(self):
        return [self.mr(mapper=self.mapper1,
                        reducer=self.reducer1),
                
                self.mr(reducer=self.reducer2)
               
               ]

if __name__ == '__main__':
    kldivergence.run()
```

    Overwriting kldivergence.py



```python
%reload_ext autoreload
%autoreload 2
from mrjob.job import MRJob
from kldivergence import kldivergence
#import numpy as np

#dont forget to save kltext.txt (see earlier cell)
mr_job = kldivergence(args=['kltext.txt'])
with mr_job.make_runner() as runner: 
    runner.run()
    # stream_output: get access of the output 
    for line in runner.stream_output():
        print mr_job.parse_output_line(line)
```

    ('KLDivergence', 0.08088278445318145)


Questions:

### MT7. Which number below is the closest to the result you get for KLD(Line1||line2)?   
(a) 0.7   
(b) 0.5   
(c) 0.2   
(d) 0.1  

d) it's a stretch

### MT8. Which of the following letters are missing from these character vectors?   
(a) p and t   
(b) k and q   
(c) j and q   
(d) j and f   


```python
import re
#) we could use the whole text as a single vector here, because if it's in one of the vector, it would influence KL distance
#) I think it's more interesting to check the one that is missing in one vector but present in the other
#) that is going to influence the most
line ="Data Science is an interdisciplinary field about processes and systems to extract knowledge or insights from large volumes of data in various forms (data in various forms, data in various forms, data in various forms), either structured or unstructured,[1][2] which is a continuation of some of the data analysis fields such as statistics, data mining and predictive analytics, as well as Knowledge Discovery in Databases.Machine learning is a subfield of computer science[1] that evolved from the study of pattern recognition and computational learning theory in artificial intelligence.[1] Machine learning explores the study and construction of algorithms that can learn from and make predictions on data.[2] Such algorithms operate by building a model from example inputs in order to make data-driven predictions or decisions,rather than following strictly static program instructions."
letters_question = ['p', 'k', 'f', 'q', 'j']
letters = set(re.sub(r"[^A-Za-z]+", '', line).lower())
#print letters
for i in letters_question:
    if i not in letters:
        print i
```

    q
    j


c


```python
%matplotlib inline
from matplotlib import pyplot as mp
import pandas as pd
from pylab import rcParams
import scipy.stats as stats
import pylab as pl
import math
import numpy as np
p = pd.read_csv("p.txt", sep="\t", names =["P_Distribution"])
q = pd.read_csv("q.txt", sep="\t", names =["Q_Distribution"])
kll = p.P_Distribution*np.log(p.P_Distribution/q.Q_Distribution)

q = (q*100).round(1)
p = (p*100).round(1)
dist = pd.concat([q, p], axis=1)
rcParams['figure.figsize'] =30, 10
plt = dist.plot()
kll.plot(secondary_y=True)  ##titleless              


mp.show()
```


![png](sahab_mt_files/sahab_mt_31_0.png)



```python

rcParams['figure.figsize'] =30, 10
#dist = pd.concat([q, p], axis=1)
kll_qtp = q.Q_Distribution*np.log(q.Q_Distribution/p.P_Distribution)
plt2 = dist.plot()
kll_qtp.plot(secondary_y=True)
mp.show()
```


![png](sahab_mt_files/sahab_mt_32_0.png)



```python
%%writefile kldivergence_smooth.py
from __future__ import division
from mrjob.job import MRJob
import re
import numpy as np
class kldivergence_smooth(MRJob):
    
    # process each string character by character
    # the relative frequency of each character emitting Pr(character|str)
    # for input record 1.abcbe
    # emit "a"    [1, (1+1)/(5+24)]
    # emit "b"    [1, (2+1)/(5+24) etc...
    def mapper1(self, _, line):
        index = int(line.split('.',1)[0])
        letter_list = re.sub(r"[^A-Za-z]+", '', line).lower()
        count = {}
        
        # (ni+1)/(n+24)
        
        for l in letter_list:
            if count.has_key(l):
                count[l] += 1
            else:
                count[l] = 1
        for key in count:
            #yield key, [index, count[key]*1.0/len(letter_list)]
            yield key, [index, (count[key]+1.0 )*1.0/(len(letter_list)+24)]

    
    def reducer1(self, key, values):
        p = 0
        q = 0
        for v in values:
            if v[0] == 1:
                p = v[1]
            else:
                q = v[1]
                
        yield "*",  p*np.log(p/q)

    # Aggregate components             
    def reducer2(self, key, values):
        kl_sum = 0
        for value in values:
            kl_sum = kl_sum + value
        yield "KLDivergence", kl_sum
            
    def steps(self):
        return [self.mr(mapper=self.mapper1,
                        reducer=self.reducer1),
                self.mr(reducer=self.reducer2)
               
               ]

if __name__ == '__main__':
    kldivergence_smooth.run()
```

    Overwriting kldivergence_smooth.py



```python
%reload_ext autoreload
%autoreload 2

from kldivergence_smooth import kldivergence_smooth
mr_job = kldivergence_smooth(args=['kltext.txt'])
with mr_job.make_runner() as runner: 
    runner.run()
    # stream_output: get access of the output 
    for line in runner.stream_output():
        print mr_job.parse_output_line(line)
```

    ('KLDivergence', 0.06726997279170038)


### MT9. The KL divergence on multinomials is defined only when they have nonzero entries. 
For zero entries, we have to smooth distributions. Suppose we smooth in this way:    

(ni+1)/(n+24)   

where ni is the count for letter i and n is the total count of all letters.    
After smoothing, which number below is the closest to the result you get for KLD(Line1||line2)??   

(a) 0.08   
(b) 0.71     
(c) 0.02    
(d) 0.11   



A

### MT10. Block size, and mapper tasks
Given ten (10) files in the input directory for a Hadoop Streaming job (MRjob or just Hadoop) with the following filesizes (in megabytes): 1, 2,3,4,5,6,7,8,9,10; and a block size of 5M (NOTE: normally we should set the blocksize to 1 GigB using modern computers). How many map tasks will result from processing the data in the input directory? Select the closest number from the following list.

 (a) 1 map task  
 (b) 14  
 (c) 12   
 (d) None of the above 

C)
#http://stackoverflow.com/questions/17459113/how-the-data-is-split-in-hadoop

### MT11. Aggregation
Given a purchase transaction log file where each purchase transaction contains the customer identifier, item purchased and much more information about the transaction. Which of the following statements are true about a MapReduce job that performs an  “aggregation” such as get the number of transaction per customer.

__Statements__
* (I) A mapper only job will not suffice, as each map tast only gets to see a subset of the data (e.g., one block). As such a mapper only job will only produce intermediate tallys for each customer. 
* (II) A reducer only job will suffice and is most efficient computationally
* (III) If the developer provides a  Mapper and Reducer it can potentially be more efficient than option II
* (IV) A reducer only job with a custom partitioner will suffice.

Select the most correct option from the following:

* (a) I, II, III, IV
* (b) II, IV
* (c) III, IV
* (d) III

D) I don't understand number IV how that will be applied in this situation

### MT12. Naive Bayes
Which of the following statements are true regarding Naive Bayes?

__Statements__
* (I) Naive Bayes is a machine learning algorithm that can be used for classifcation problems only
* (II) Multinomial Naive Bayes is a flavour of Naive Bayes for discrete input variables and can be combined  with Laplace smoothing to avoid zero predictions for class posterior probabilities  when attribute value combinations show up during classification but were not present during training. 
* (III) Naive Bayes can be used for continous valued input variables. In this case, one can use Gaussian distributions to model the class conditional probability distributions Pr(X|Class).
* (IV) Naive Bayes can model continous target variables directly.

Please select the single most correct combination from the following:
    

* (a) I, II, III, IV
* (b) I, II, III
* (c) I, III, IV
* (d) I, II


B) # hard to classify continuou target variable directly

### MT13. Naive Bayes SPAM model
Given the following document dataset for a  Two-Class problem: ham and spam. Use MRJob (please include your code) to build a muiltnomial Naive Bayes classifier. Please use Laplace Smoothing with a hyperparameter of 1. Please use words only (a-z) as features. Please lowercase all words.


```python
%%writefile nbtrain.txt
ham d1: "good
ham d2: "very good
spam d3: "bad
spam d4: "very bad
spam d5: "very bad, very BAD."
```

    Writing nbtrain.txt

%%writefile nbtext.txt
? d6: “good? bad! very Bad!” 

```python
%%writefile naive_bayes.py
from __future__ import division
from mrjob.job import MRJob
from collections import defaultdict
import re

class naive_bayes(MRJob):
    
    def mapper(self, _, line):
        
        #label, id, words = line.split()
        info, words = line.split(":")
        label = info.split()[0]
       # print label
        words = words.split()
       #print words
        wordcount = defaultdict(int)
        for word in words: #re.findall(r"[\w']+", words):
            word = re.sub(r'[^\x00-\x7F]+','',word)
            word = word.strip().lower()
           # word = re.findall(r"[\w']+", word)
            wordcount[word] +=1
        for word in wordcount:
            yield label, [word, wordcount[word]]
    
    def reducer(self, label, values):
        p_word_ham = defaultdict(int)
        p_word_spam = defaultdict(int)
        spamWords = defaultdict(int)
        hamWords = defaultdict(int)
        spamCount = 0
        hamCount = 0
        vocab=set()
        
        for val in values:
            word, count = val
            vocab.add(word)
            if label == "spam":
                spamWords[word] += int(count)
                spamCount += 1
            else:
                hamWords[word] += int(count)
                hamCount +=1
                
        for word in vocab:
            #will give 0 for viagara if it was never there in ham
            #print "k"
            if(spamCount != 0 & hamCount != 0): #just in case, you never know4
                #print "k"
                p_word_spam[word] = (spamWords[word]*1.00/spamCount*1.0)
                p_word_ham[word] = hamWords[word]*1.0000/hamCount*1.0

            print word, "\t", p_word_ham[word], "\t", p_word_spam[word] #, "\t", p_word_ham[word]
        
            
    def steps(self):
        return [self.mr(mapper=self.mapper, reducer=self.reducer)]

if __name__ == '__main__':
    naive_bayes.run()
```

    Overwriting naive_bayes.py



```python
from naive_bayes import naive_bayes
mr_job = naive_bayes(args=['nbtrain.txt'])
with mr_job.make_runner() as runner: 
    runner.run()
    # stream_output: get access of the output 
    for line in runner.stream_output():
        print mr_job.parse_output_line(line)


```

    very 	0 	0
    good. 	0 	0
    very 	0 	0
    bad. 	0 	0
    bad, 	0 	0



```python
#? d6: “good? bad! very Bad!
p_h = .2*.25
p_s = .25 *.5
print p_h, p_s
```

    0.05 0.125


__QUESTION__

Having learnt the Naive Bayes text classification model for this problem using the training data and classified the test data (d6) please indicate which of the following is true:

__Statements__
* (I)  P(very|ham) = 0.33
* (II) P(good|ham) = 0.50
* (I)  Posterior Probability P(ham| d6) is approximately 24%
* (IV) Class of d6 is ham

Please select the single most correct combination of these statements from the following:
    

* (a) I, II, III, IV
* (b) I, II, III
* (c) I, III, IV
* (d) I, II

C) #without smoothing

### MT14. Is there a map input format (for Hadoop or MRJob)?   

(a)  Yes, but only in Hadoop 0.22+.   
(b)  Yes, in Hadoop there is a default expectation that each record is delimited by an end of line charcacter and that key is the first token delimited by a tab character and that the value-part  is everything after the tab character.   
(c)  No,  when  MRJob INPUT_PROTOCOL = RawValueProtocol. In this case input is processed in format agnostic way thereby avoiding any type of parsing errors. The value is treated as a str, the key is read in as None.   
(d) Both b and c are correct answers.  

#D #double check

### MT15. What happens if mapper output does not match reducer input?   
 
(a)  Hadoop API will convert the data to the type that is needed by the reducer.    
(b)  Data input/output inconsistency cannot occur. A preliminary validation check is executed prior to the full execution of the job to ensure there is consistency.    
(c)  The java compiler will report an error during compilation but the job will complete with exceptions.    
(d)  A real-time exception will be thrown and map-reduce job will fail.

A

### MT16. Why would a developer create a map-reduce without the reduce step?   
 
(a)  Developers should design Map-Reduce jobs without reducers only if no reduce slots are available on the cluster.    
(b)  Developers should never design Map-Reduce jobs without reducers. An error will occur upon compile.    
(c)  There is a CPU intensive step that occurs between the map and reduce steps. Disabling the reduce step speeds up data processing.  
(d)  It is not possible to create a map-reduce job without at least one reduce step. A developer may decide to limit to one reducer for debugging purposes.   



C 

===Gradient descent===
### MT17. Which of the following are true statements with respect to gradient descent for machine learning, where alpha is the learning rate. Select all that apply

* (I) To make gradient descent converge, we must slowly decrease alpha over time and use a combiner in the context of Hadoop.
* (II) Gradient descent is guaranteed to find the global minimum for any unconstrained convex objective function J() regardless of using a combiner or not in the context of Hadoop
* (III) Gradient descent can converge even if alpha is kept fixed. (But alpha cannot be too large, or else it may fail to converge.) Combiners will help speed up the process.
* (IV) For the specific choice of cost function J() used in linear regression, there is no local optima (other than the global optimum).


Select a single correct response from the following:
* (a) I, II, III, IV
* (b) I, III, IV
* (c) II, III
* (d) II,III, IV

D

===Weighted K-means===

Write a MapReduce job in MRJob to do the training at scale of a weighted K-means algorithm.

You can write your own code or you can use most of the code from the following notebook:

* http://nbviewer.jupyter.org/urls/dl.dropbox.com/s/oppgyfqxphlh69g/MrJobKmeans_Corrected.ipynb

Weight each example as follows using the inverse vector length (Euclidean norm): 

weight(X)= 1/||X||, 

where ||X|| = SQRT(X.X)= SQRT(X1^2 + X2^2)

Here X is vector made up of two component X1 and X2.

Using the following data to answer the following TWO questions:

* https://www.dropbox.com/s/ai1uc3q2ucverly/Kmeandata.csv?dl=0

### MT18. Which result below is the closest to the centroids you got after running your weighted K-means code for K=3 for 10 iterations? 
(old11-12)

* (a) (-4.0,0.0), (4.0,0.0), (6.0,6.0)     
* (b) (-4.5,0.0), (4.5,0.0), (0.0,4.5)    
* (c) (-5.5,0.0), (0.0,0.0), (3.0,3.0)    
* (d) (-4.5,0.0), (-4.0,0.0), (0.0,4.5)  

B) Counldn't figure out with new KLL used regular KNN'


```python
!wget - q -O kmeans.csv https://www.dropbox.com/s/ai1uc3q2ucverly/Kmeandata.csv?dl=0
```

    --2016-10-23 14:13:15--  http://-/
    Resolving -... failed: Name or service not known.
    wget: unable to resolve host address “-”
    --2016-10-23 14:13:15--  http://q/
    Resolving q... failed: Name or service not known.
    wget: unable to resolve host address “q”
    --2016-10-23 14:13:15--  https://www.dropbox.com/s/ai1uc3q2ucverly/Kmeandata.csv?dl=0
    Resolving www.dropbox.com... 162.125.4.1
    Connecting to www.dropbox.com|162.125.4.1|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://dl.dropboxusercontent.com/content_link/cTf99c1qujXFgxbXrG5vhrAWm43OeajdLRPtxALGUDOtUwRekfdFhgPWamISjYbW/file [following]
    --2016-10-23 14:13:16--  https://dl.dropboxusercontent.com/content_link/cTf99c1qujXFgxbXrG5vhrAWm43OeajdLRPtxALGUDOtUwRekfdFhgPWamISjYbW/file
    Resolving dl.dropboxusercontent.com... 45.58.74.5
    Connecting to dl.dropboxusercontent.com|45.58.74.5|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 155486 (152K) [text/csv]
    Saving to: “kmeans.csv”
    
    100%[======================================>] 155,486     --.-K/s   in 0.08s   
    
    2016-10-23 14:13:17 (1.92 MB/s) - “kmeans.csv” saved [155486/155486]
    
    FINISHED --2016-10-23 14:13:17--
    Downloaded: 1 files, 152K in 0.08s (1.92 MB/s)



```python
%%writefile Kmeans.py
from numpy import argmin, array, random
from mrjob.job import MRJob
from mrjob.step import MRStep
from itertools import chain
import os

#Calculate find the nearest centroid for data point 
def MinDist(datapoint, centroid_points):
    datapoint = array(datapoint)
    centroid_points = array(centroid_points)
    diff = datapoint - centroid_points 
    diffsq = diff*diff
    # Get the nearest centroid for each instance
    minidx = argmin(list(diffsq.sum(axis = 1)))

    return minidx

#Check whether centroids converge
def stop_criterion(centroid_points_old, centroid_points_new,T):
    oldvalue = list(chain(*centroid_points_old))
    newvalue = list(chain(*centroid_points_new))
    Diff = [abs(x-y) for x, y in zip(oldvalue, newvalue)]
    Flag = True
    for i in Diff:
        if(i>T):
            Flag = False
            break
    return Flag
def listsum(numList):
        theSum = 0
        for i in numList:
            theSum = theSum + i
        return theSum

class MRKmeans(MRJob):
    centroid_points=[]
    k=3    
    def steps(self):
        return [
            MRStep(mapper_init = self.mapper_init, mapper=self.mapper,combiner = self.combiner,reducer=self.reducer)
               ]
    #load centroids info from file
    def mapper_init(self):
        print "Current path:", os.path.dirname(os.path.realpath(__file__))
        
        self.centroid_points = [map(float,s.split('\n')[0].split(',')) for s in open("Centroids.txt").readlines()]
        #open('Centroids.txt', 'w').close()
        
        print "Centroids: ", self.centroid_points
        
    #load data and output the nearest centroid index and data point 
    def mapper(self, _, line):
        D = (map(float,line.split(',')))
        yield int(MinDist(D,self.centroid_points)), (D[0],D[1],1)
        
    #Combine sum of data points locally
    def combiner(self, idx, inputdata):
        sumx = sumy = num = 0
        for x,y,n in inputdata:
            num = num + n
            sumx = sumx + x
            sumy = sumy + y
        yield idx,(sumx,sumy,num)
    #Aggregate sum for each cluster and then calculate the new centroids
    #
    
    

    def reducer(self, idx, inputdata): 
        centroids = []
        num = [0]*self.k 
        total_weight = [0]*self.k 
        w = 0
        #norm = 1.0/(datapoint**2 + centroid_points**2 )**.5
        for i in range(self.k):
            centroids.append([0,0])
            
        ####Add weight in deciding the centroids###
        for x, y, n in inputdata:
            #points weight
            #we must take norm with respect to centroid otherwise 
            #centroids would be pulled towards zero/origin instead of
            #the dense part of the cluster
            weight = 1.0/((centroids[idx][0]-x)**2 + (centroids[idx][1]-y)**2 )**.5
           # weight = 1.0/((x)**2 + (y)**2 )**.5
            w = w + weight
            num[idx] = num[idx] + n
            total_weight[idx] = total_weight[idx] + weight # is giving [0, 0.0008, 0]
            centroids[idx][0] = centroids[idx][0] + x*weight
            centroids[idx][1] = centroids[idx][1] + y*weight
           # total_weight += weight
        #print "weih", total_weight, w*num[idx]
        print w
        normalizer = (w/num[idx])
        centroids[idx][0] = centroids[idx][0]/(num[idx])
        centroids[idx][1] = centroids[idx][1]/(num[idx])
        yield idx,(centroids[idx][0],centroids[idx][1])
      
if __name__ == '__main__':
    MRKmeans.run()
```

    Overwriting Kmeans.py



```python
%reload_ext autoreload
%autoreload 2
from numpy import random
from Kmeans import MRKmeans, stop_criterion
mr_job = MRKmeans(args=['/home/cloudera/kmeans.csv', '--file=Centroids.txt'])

#Geneate initial centroids
centroid_points = []
k = 3
for i in range(k):
    centroid_points.append([random.uniform(-3,3),random.uniform(-3,3)])
with open('Centroids.txt', 'w+') as f:
        f.writelines(','.join(str(j) for j in i) + '\n' for i in centroid_points)

# Update centroids iteratively
i = 0
while(1):
    # save previous centoids to check convergency
    centroid_points_old = centroid_points[:]
    print "iteration"+str(i)+":"
    with mr_job.make_runner() as runner: 
        runner.run()
        # stream_output: get access of the output 
        for line in runner.stream_output():
            key,value =  mr_job.parse_output_line(line)
            print key, value
            centroid_points[key] = value
            
        # Update the centroids for the next iteration
        with open('Centroids.txt', 'w') as f:
            f.writelines(','.join(str(j) for j in i) + '\n' for i in centroid_points)
        
    print "\n"
    i = i + 1
    if(i==10):
   # if(stop_criterion(centroid_points_old,centroid_points,0.01)):
        break
print "Centroids\n"
print centroid_points
```

    iteration0:
    Current path: /tmp/Kmeans.cloudera.20161023.213139.012570/job_local_dir/0/mapper/0
    Centroids:  [[-2.10625857251, -0.87191201499], [-2.92073595192, -0.174421775216], [1.50007058092, -1.88737986427]]
    Current path: /tmp/Kmeans.cloudera.20161023.213139.012570/job_local_dir/0/mapper/1
    Centroids:  [[-2.10625857251, -0.87191201499], [-2.92073595192, -0.174421775216], [1.50007058092, -1.88737986427]]
    0.0395276037036
    0.000589920755662
    0.000761724368606
    0 [-0.030559197357327223, 0.0009950268838383163]
    1 [-0.0008187159686352959, 0.0007285554240286462]
    2 [0.0017969963529232507, 0.00015564533841940672]
    
    
    iteration1:
    Current path: /tmp/Kmeans.cloudera.20161023.213139.289008/job_local_dir/0/mapper/0
    Centroids:  [[-0.0305591973573, 0.000995026883838], [-0.000818715968635, 0.000728555424029], [0.00179699635292, 0.000155645338419]]
    Current path: /tmp/Kmeans.cloudera.20161023.213139.289008/job_local_dir/0/mapper/1
    Centroids:  [[-0.0305591973573, 0.000995026883838], [-0.000818715968635, 0.000728555424029], [0.00179699635292, 0.000155645338419]]
    0.000667498492197
    0.00258043241634
    0.000731427613415
    0 [-0.001231854318891861, 0.000547059738201556]
    1 [0.0007202846939335756, 0.006587093694713488]
    2 [0.0016200615705719934, 0.0002980290659668441]
    
    
    iteration2:
    Current path: /tmp/Kmeans.cloudera.20161023.213139.539584/job_local_dir/0/mapper/0
    Centroids:  [[-0.00123185431889, 0.000547059738202], [0.000720284693934, 0.00658709369471], [0.00162006157057, 0.000298029065967]]
    Current path: /tmp/Kmeans.cloudera.20161023.213139.539584/job_local_dir/0/mapper/1
    Centroids:  [[-0.00123185431889, 0.000547059738202], [0.000720284693934, 0.00658709369471], [0.00162006157057, 0.000298029065967]]
    0.000888530588623
    0.00069860001744
    0.00109933709277
    0 [-0.0022574227251754516, -0.00011379456752410971]
    1 [0.00023683207532860476, 0.0014068976332962735]
    2 [0.002783802362894958, -0.0003500615560259285]
    
    
    iteration3:
    Current path: /tmp/Kmeans.cloudera.20161023.213139.796857/job_local_dir/0/mapper/0
    Centroids:  [[-0.00225742272518, -0.000113794567524], [0.000236832075329, 0.0014068976333], [0.00278380236289, -0.000350061556026]]
    Current path: /tmp/Kmeans.cloudera.20161023.213139.796857/job_local_dir/0/mapper/1
    Centroids:  [[-0.00225742272518, -0.000113794567524], [0.000236832075329, 0.0014068976333], [0.00278380236289, -0.000350061556026]]
    0.00079446539021
    0.000820448375324
    0.000793707903886
    0 [-0.0019627878886563886, 2.3544482030906706e-05]
    1 [2.4148238918362396e-05, 0.002068509919081527]
    2 [0.001972728242351702, 8.738180464415093e-06]
    
    
    iteration4:
    Current path: /tmp/Kmeans.cloudera.20161023.213140.043150/job_local_dir/0/mapper/0
    Centroids:  [[-0.00196278788866, 2.35444820309e-05], [2.41482389184e-05, 0.00206850991908], [0.00197272824235, 8.73818046442e-06]]
    Current path: /tmp/Kmeans.cloudera.20161023.213140.043150/job_local_dir/0/mapper/1
    Centroids:  [[-0.00196278788866, 2.35444820309e-05], [2.41482389184e-05, 0.00206850991908], [0.00197272824235, 8.73818046442e-06]]
    0.000802170374024
    0.000801490016964
    0.000800442268879
    0 [-0.002000343672702529, -4.5348484330416053e-07]
    1 [2.064198769695743e-05, 0.001994279534569277]
    2 [0.0020063556358775334, -1.1462988645188453e-05]
    
    
    iteration5:
    Current path: /tmp/Kmeans.cloudera.20161023.213140.283990/job_local_dir/0/mapper/0
    Centroids:  [[-0.0020003436727, -4.53484843304e-07], [2.0641987697e-05, 0.00199427953457], [0.00200635563588, -1.14629886452e-05]]
    Current path: /tmp/Kmeans.cloudera.20161023.213140.283990/job_local_dir/0/mapper/1
    Centroids:  [[-0.0020003436727, -4.53484843304e-07], [2.0641987697e-05, 0.00199427953457], [0.00200635563588, -1.14629886452e-05]]
    0.000801741181997
    0.000802250205529
    0.000800001834743
    0 [-0.0019983376395327082, 5.711927775000548e-07]
    1 [2.085578146111719e-05, 0.00199824679154487]
    2 [0.002004350950006126, -1.0463122071513352e-05]
    
    
    iteration6:
    Current path: /tmp/Kmeans.cloudera.20161023.213140.516523/job_local_dir/0/mapper/0
    Centroids:  [[-0.00199833763953, 5.711927775e-07], [2.08557814611e-05, 0.00199824679154], [0.00200435095001, -1.04631220715e-05]]
    Current path: /tmp/Kmeans.cloudera.20161023.213140.516523/job_local_dir/0/mapper/1
    Centroids:  [[-0.00199833763953, 5.711927775e-07], [2.08557814611e-05, 0.00199824679154], [0.00200435095001, -1.04631220715e-05]]
    0.000801741181997
    0.000802250205529
    0.000800001834743
    0 [-0.0019983376395327082, 5.711927775000548e-07]
    1 [2.085578146111719e-05, 0.00199824679154487]
    2 [0.002004350950006126, -1.0463122071513352e-05]
    
    
    iteration7:
    Current path: /tmp/Kmeans.cloudera.20161023.213140.765117/job_local_dir/0/mapper/0
    Centroids:  [[-0.00199833763953, 5.711927775e-07], [2.08557814611e-05, 0.00199824679154], [0.00200435095001, -1.04631220715e-05]]
    Current path: /tmp/Kmeans.cloudera.20161023.213140.765117/job_local_dir/0/mapper/1
    Centroids:  [[-0.00199833763953, 5.711927775e-07], [2.08557814611e-05, 0.00199824679154], [0.00200435095001, -1.04631220715e-05]]
    0.000801741181997
    0.000802250205529
    0.000800001834743
    0 [-0.0019983376395327082, 5.711927775000548e-07]
    1 [2.085578146111719e-05, 0.00199824679154487]
    2 [0.002004350950006126, -1.0463122071513352e-05]
    
    
    iteration8:
    Current path: /tmp/Kmeans.cloudera.20161023.213140.999843/job_local_dir/0/mapper/0
    Centroids:  [[-0.00199833763953, 5.711927775e-07], [2.08557814611e-05, 0.00199824679154], [0.00200435095001, -1.04631220715e-05]]
    Current path: /tmp/Kmeans.cloudera.20161023.213140.999843/job_local_dir/0/mapper/1
    Centroids:  [[-0.00199833763953, 5.711927775e-07], [2.08557814611e-05, 0.00199824679154], [0.00200435095001, -1.04631220715e-05]]
    0.000801741181997
    0.000802250205529
    0.000800001834743
    0 [-0.0019983376395327082, 5.711927775000548e-07]
    1 [2.085578146111719e-05, 0.00199824679154487]
    2 [0.002004350950006126, -1.0463122071513352e-05]
    
    
    iteration9:
    Current path: /tmp/Kmeans.cloudera.20161023.213141.273174/job_local_dir/0/mapper/0
    Centroids:  [[-0.00199833763953, 5.711927775e-07], [2.08557814611e-05, 0.00199824679154], [0.00200435095001, -1.04631220715e-05]]
    Current path: /tmp/Kmeans.cloudera.20161023.213141.273174/job_local_dir/0/mapper/1
    Centroids:  [[-0.00199833763953, 5.711927775e-07], [2.08557814611e-05, 0.00199824679154], [0.00200435095001, -1.04631220715e-05]]
    0.000801741181997
    0.000802250205529
    0.000800001834743
    0 [-0.0019983376395327082, 5.711927775000548e-07]
    1 [2.085578146111719e-05, 0.00199824679154487]
    2 [0.002004350950006126, -1.0463122071513352e-05]
    
    
    Centroids
    
    [[-0.0019983376395327082, 5.711927775000548e-07], [2.085578146111719e-05, 0.00199824679154487], [0.002004350950006126, -1.0463122071513352e-05]]


### MT19. Using the result of the previous question, which number below is the closest  to the average weighted distance between each example and its assigned (closest) centroid?

The average weighted distance is defined as 
sum over i  (weighted_distance_i)     /  sum over i (weight_i)


* (a) 2.5     
* (b) 1.5     
* (c) 0.5     
* (d) 4.0   

b) Picking the lowest because it is closed to total weight,


```python
%%writefile Kmeans.py
from numpy import argmin, array, random
from mrjob.job import MRJob
from mrjob.step import MRStep
from itertools import chain
import os

#Calculate find the nearest centroid for data point 
def MinDist(datapoint, centroid_points):
    datapoint = array(datapoint)
    centroid_points = array(centroid_points)
    diff = datapoint - centroid_points 
    diffsq = diff*diff
    # Get the nearest centroid for each instance
    minidx = argmin(list(diffsq.sum(axis = 1)))
    return minidx

#Check whether centroids converge
def stop_criterion(centroid_points_old, centroid_points_new,T):
    oldvalue = list(chain(*centroid_points_old))
    newvalue = list(chain(*centroid_points_new))
    Diff = [abs(x-y) for x, y in zip(oldvalue, newvalue)]
    Flag = True
    for i in Diff:
        if(i>T):
            Flag = False
            break
    return Flag

class MRKmeans(MRJob):
    centroid_points=[]
    k=3    
    def steps(self):
        return [
            MRStep(mapper_init = self.mapper_init, mapper=self.mapper,combiner = self.combiner,reducer=self.reducer)
               ]
    #load centroids info from file
    def mapper_init(self):
        print "Current path:", os.path.dirname(os.path.realpath(__file__))
        
        self.centroid_points = [map(float,s.split('\n')[0].split(',')) for s in open("Centroids.txt").readlines()]
        #open('Centroids.txt', 'w').close()
        
        print "Centroids: ", self.centroid_points
        
    #load data and output the nearest centroid index and data point 
    def mapper(self, _, line):
        D = (map(float,line.split(',')))
        yield int(MinDist(D,self.centroid_points)), (D[0],D[1],1)
    #Combine sum of data points locally
    def combiner(self, idx, inputdata):
        sumx = sumy = num = 0
        for x,y,n in inputdata:
            num = num + n
            sumx = sumx + x
            sumy = sumy + y
        yield idx,(sumx,sumy,num)
    #Aggregate sum for each cluster and then calculate the new centroids
    def reducer(self, idx, inputdata): 
        centroids = []
        num = [0]*self.k 
        for i in range(self.k):
            centroids.append([0,0])
        for x, y, n in inputdata:
            num[idx] = num[idx] + n
            centroids[idx][0] = centroids[idx][0] + x
            centroids[idx][1] = centroids[idx][1] + y
        centroids[idx][0] = centroids[idx][0]/num[idx]
        centroids[idx][1] = centroids[idx][1]/num[idx]

        yield idx,(centroids[idx][0],centroids[idx][1])
      
if __name__ == '__main__':
    MRKmeans.run()
```

    Overwriting Kmeans.py



```python
%reload_ext autoreload
%autoreload 2
from numpy import random
from Kmeans import MRKmeans, stop_criterion
mr_job = MRKmeans(args=['kmeans.csv', '--file=Centroids.txt'])

#Geneate initial centroids
centroid_points = []
k = 3
for i in range(k):
    centroid_points.append([random.uniform(-3,3),random.uniform(-3,3)])
with open('Centroids.txt', 'w+') as f:
        f.writelines(','.join(str(j) for j in i) + '\n' for i in centroid_points)

# Update centroids iteratively
i = 0
while(1):
    # save previous centoids to check convergency
    centroid_points_old = centroid_points[:]
    print "iteration"+str(i)+":"
    with mr_job.make_runner() as runner: 
        runner.run()
        # stream_output: get access of the output 
        for line in runner.stream_output():
            key,value =  mr_job.parse_output_line(line)
            print key, value
            centroid_points[key] = value
            
        # Update the centroids for the next iteration
        with open('Centroids.txt', 'w') as f:
            f.writelines(','.join(str(j) for j in i) + '\n' for i in centroid_points)
        
    print "\n"
    i = i + 1
    if(stop_criterion(centroid_points_old,centroid_points,0.01)):
        break
print "Centroids\n"
print centroid_points
```

    iteration0:
    Current path: /tmp/Kmeans.cloudera.20161023.213009.408667/job_local_dir/0/mapper/0
    Centroids:  [[-1.75312412735, -0.522128768623], [0.264628516987, 1.12269395623], [-2.50001824328, -0.505566871974]]
    Current path: /tmp/Kmeans.cloudera.20161023.213009.408667/job_local_dir/0/mapper/1
    Centroids:  [[-1.75312412735, -0.522128768623], [0.264628516987, 1.12269395623], [-2.50001824328, -0.505566871974]]
    0 [-1.7904060067059906, 0.15809043794498376]
    1 [2.5517749819686366, 2.482631469627905]
    2 [-5.0293301603928535, 0.009883066682740235]
    
    
    iteration1:
    Current path: /tmp/Kmeans.cloudera.20161023.213009.687007/job_local_dir/0/mapper/0
    Centroids:  [[-1.79040600671, 0.158090437945], [2.55177498197, 2.48263146963], [-5.02933016039, 0.00988306668274]]
    Current path: /tmp/Kmeans.cloudera.20161023.213009.687007/job_local_dir/0/mapper/1
    Centroids:  [[-1.79040600671, 0.158090437945], [2.55177498197, 2.48263146963], [-5.02933016039, 0.00988306668274]]
    0 [-2.205858707618006, 2.3127798862284714]
    1 [2.8558403359006412, 2.374765514609414]
    2 [-5.258484914061264, -0.028792236626079996]
    
    
    iteration2:
    Current path: /tmp/Kmeans.cloudera.20161023.213009.896535/job_local_dir/0/mapper/0
    Centroids:  [[-2.20585870762, 2.31277988623], [2.8558403359, 2.37476551461], [-5.25848491406, -0.0287922366261]]
    Current path: /tmp/Kmeans.cloudera.20161023.213009.896535/job_local_dir/0/mapper/1
    Centroids:  [[-2.20585870762, 2.31277988623], [2.8558403359, 2.37476551461], [-5.25848491406, -0.0287922366261]]
    0 [-1.1917136455618718, 4.4422786351744294]
    1 [3.8516618902288875, 1.514605528675241]
    2 [-5.18946637665841, -0.1365978763972875]
    
    
    iteration3:
    Current path: /tmp/Kmeans.cloudera.20161023.213010.165509/job_local_dir/0/mapper/0
    Centroids:  [[-1.19171364556, 4.44227863517], [3.85166189023, 1.51460552868], [-5.18946637666, -0.136597876397]]
    Current path: /tmp/Kmeans.cloudera.20161023.213010.165509/job_local_dir/0/mapper/1
    Centroids:  [[-1.19171364556, 4.44227863517], [3.85166189023, 1.51460552868], [-5.18946637666, -0.136597876397]]
    0 [-0.0979067726217919, 5.003614134729571]
    1 [4.929135752834473, 0.14099509575286046]
    2 [-5.013904536695912, -0.02842293373227361]
    
    
    iteration4:
    Current path: /tmp/Kmeans.cloudera.20161023.213010.361904/job_local_dir/0/mapper/0
    Centroids:  [[-0.0979067726218, 5.00361413473], [4.92913575283, 0.140995095753], [-5.0139045367, -0.0284229337323]]
    Current path: /tmp/Kmeans.cloudera.20161023.213010.361904/job_local_dir/0/mapper/1
    Centroids:  [[-0.0979067726218, 5.00361413473], [4.92913575283, 0.140995095753], [-5.0139045367, -0.0284229337323]]
    0 [0.04009229720112095, 4.994383121818837]
    1 [5.030614702678463, -0.015280450423949538]
    2 [-4.988208799410262, -0.0016052213546992817]
    
    
    iteration5:
    Current path: /tmp/Kmeans.cloudera.20161023.213010.656224/job_local_dir/0/mapper/0
    Centroids:  [[0.0400922972011, 4.99438312182], [5.03061470268, -0.0152804504239], [-4.98820879941, -0.0016052213547]]
    Current path: /tmp/Kmeans.cloudera.20161023.213010.656224/job_local_dir/0/mapper/1
    Centroids:  [[0.0400922972011, 4.99438312182], [5.03061470268, -0.0152804504239], [-4.98820879941, -0.0016052213547]]
    0 [0.053065423788147964, 4.987793423944292]
    1 [5.0402327160888465, -0.026294229978289455]
    2 [-4.98580568889943, 0.0009376094363626959]
    
    
    iteration6:
    Current path: /tmp/Kmeans.cloudera.20161023.213010.869481/job_local_dir/0/mapper/0
    Centroids:  [[0.0530654237881, 4.98779342394], [5.04023271609, -0.0262942299783], [-4.9858056889, 0.000937609436363]]
    Current path: /tmp/Kmeans.cloudera.20161023.213010.869481/job_local_dir/0/mapper/1
    Centroids:  [[0.0530654237881, 4.98779342394], [5.04023271609, -0.0262942299783], [-4.9858056889, 0.000937609436363]]
    0 [0.053065423788147964, 4.987793423944292]
    1 [5.0402327160888465, -0.026294229978289455]
    2 [-4.98580568889943, 0.0009376094363626959]
    
    
    Centroids
    
    [[0.053065423788147964, 4.987793423944292], [5.0402327160888465, -0.026294229978289455], [-4.98580568889943, 0.0009376094363626959]]



```python
%%writefile kldivergence2.py
# coding: utf-8
#code to produce distrbutions for KLL
#didn't have time to add this in the original code

from __future__ import division
from mrjob.job import MRJob
from mrjob.step import MRStep
import re
import numpy as np

class kldivergence2(MRJob):
    # process each string character by character
    # the relative frequency of each character emitting Pr(character|str)
    # for input record 1.abcbe
    # emit "a"    [1, 0.2]
    # emit "b"    [1, 0.4] etc...
    def mapper1(self, _, line):
        index = int(line.split('.',1)[0])
        letter_list = re.sub(r"[^A-Za-z]+", '', line).lower()
        count = {}
        for l in letter_list:
            if count.has_key(l):
                count[l] += 1
            else:
                count[l] = 1
        for key in count:
            yield key, [index, count[key]*1.0/len(letter_list)]

    # on a component i calculate (e.g., "b")
    # Kullback–Leibler divergence of Q from P is defined as (P(i) log (P(i) / Q(i))
    def reducer1(self, key, values):
        
      #  print key
        p = 0
        q = 0
        #toggle manually
        #could write both files here
        for v in values:
            if v[0] == 1:  #String 1
                p = v[1]
                print p
            else:          # String 2
                q = v[1]
                #print  q
            
            
    def steps(self):
        mr_steps = [self.mr(mapper=self.mapper1,
                        reducer=self.reducer1)]
#         mr_steps = [MRStep(mapper=self.mapper1, reducer=self.reducer1)]
        return mr_steps

if __name__ == '__main__':
    kldivergence2.run()
```

    Overwriting kldivergence2.py



```python

#toggle with the changes in script above
!cat kltext.txt | python  kldivergence2.py > p.txt
#!cat kltext.txt | python  kldivergence2.py > q2.txt
```

    No configs found; falling back on auto-configuration
    Creating temp directory /tmp/kldivergence2.cloudera.20161023.200017.136265
    mr() is deprecated and will be removed in v0.6.0. Use mrjob.step.MRStep directly instead.
    Running step 1 of 1...
    reading from STDIN
    mr() is deprecated and will be removed in v0.6.0. Use mrjob.step.MRStep directly instead.
    mr() is deprecated and will be removed in v0.6.0. Use mrjob.step.MRStep directly instead.
    mr() is deprecated and will be removed in v0.6.0. Use mrjob.step.MRStep directly instead.
    mr() is deprecated and will be removed in v0.6.0. Use mrjob.step.MRStep directly instead.
    mr() is deprecated and will be removed in v0.6.0. Use mrjob.step.MRStep directly instead.
    mr() is deprecated and will be removed in v0.6.0. Use mrjob.step.MRStep directly instead.
    Streaming final output from /tmp/kldivergence2.cloudera.20161023.200017.136265/output...
    Removing temp directory /tmp/kldivergence2.cloudera.20161023.200017.136265...



```python
#!cat kltext.txt | python  kldivergence2.py > p1.txt
!cat kltext.txt | python  kldivergence2.py > q.txt
```

    No configs found; falling back on auto-configuration
    Creating temp directory /tmp/kldivergence2.cloudera.20161023.195955.232193
    mr() is deprecated and will be removed in v0.6.0. Use mrjob.step.MRStep directly instead.
    Running step 1 of 1...
    reading from STDIN
    mr() is deprecated and will be removed in v0.6.0. Use mrjob.step.MRStep directly instead.
    mr() is deprecated and will be removed in v0.6.0. Use mrjob.step.MRStep directly instead.
    mr() is deprecated and will be removed in v0.6.0. Use mrjob.step.MRStep directly instead.
    mr() is deprecated and will be removed in v0.6.0. Use mrjob.step.MRStep directly instead.
    mr() is deprecated and will be removed in v0.6.0. Use mrjob.step.MRStep directly instead.
    mr() is deprecated and will be removed in v0.6.0. Use mrjob.step.MRStep directly instead.
    Streaming final output from /tmp/kldivergence2.cloudera.20161023.195955.232193/output...
    Removing temp directory /tmp/kldivergence2.cloudera.20161023.195955.232193...



```python
!cat p.txt
```

    0.110787172012
    0.00583090379009
    0.0408163265306
    0.0553935860058
    0.0758017492711
    0.0291545189504
    0.0145772594752
    0.0174927113703
    0.0962099125364
    0.00583090379009
    0.0320699708455
    0.0262390670554
    0.064139941691
    0.069970845481
    0.00874635568513
    0.067055393586
    0.110787172012
    0.0816326530612
    0.0379008746356
    0.0204081632653
    0.0116618075802
    0.00291545189504
    0.0145772594752


# END of Exam
