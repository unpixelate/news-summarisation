#%%
import sys
import os
from spacy.lang.en import English
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from summarizer.cluster import ClusterEmbeddings
from summarizer.base import BERT
from summarizer.coreference import CoreferenceHandler,debug,DEBUG
import logging
logging.basicConfig(level=logging.critical)

DEBUG = True

class Summarizer:
    def __init__(self
        ,language = English
        ,sentence_handler = CoreferenceHandler()
        ,tokenizer = None
        ,model = None
        ,hidden = -2
        ,reduce_option = 'mean'
        ,random_state = 2020
        ,small_model = False):

        self.nlp = language()
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        if not small_model:
            self.model = BERT( 'bart-large-cnn')
        self.augment_model = BERT('bert-base-uncased')
        self.hidden = hidden
        self.reduce_option = reduce_option
        self.random_state = random_state
        self.sentence_handler = sentence_handler
        self.small_model = small_model


    def process_sentence(self,text,min_length: int, max_length: int):
        text = text.replace('\n','')
        text = text.replace(':','-')
        doc = self.sentence_handler.process(text,min_length,max_length)
        return doc

    def run_cluster(self, content, ratio, algorithm, use_first):
        hidden = self.model(content, self.hidden, self.reduce_option)
        augment_values = self.augment_model(content, self.hidden, self.reduce_option)


        clusterer = ClusterEmbeddings(hidden, algorithm, random_state=self.random_state)
        augment_clusterer = ClusterEmbeddings(augment_values, algorithm, random_state=self.random_state)

        hidden_args = clusterer(ratio)
        augment_clusterer_args = augment_clusterer(ratio)
        
        voted = self.combine_sentence(hidden_args,augment_clusterer_args)

        if use_first and voted[0] != 0:
            voted.insert(0,0)
            print(voted)
        return [content[j] for j in voted]

    def fast_run_cluster(self, content, ratio, algorithm, use_first):
        small_hidden = self.augment_model(content, self.hidden, self.reduce_option)
        small_cluster = ClusterEmbeddings(small_hidden, algorithm, random_state=self.random_state)
        voted = small_cluster(ratio)
        if use_first and voted[0] != 0:
            voted.insert(0,0)
        return [content[j] for j in voted]


    def run(self, text,ratio,algorithm,use_first,min_length=None,max_length=None, fast=False):
        if min_length and max_length:
            sentences = self.process_sentence(text,min_length,max_length)
        else:
            sentences = self.process_sentence(text)
        if sentences and not self.small_model:
            sentences = self.run_cluster(sentences, ratio, algorithm, use_first)
        if sentences and self.small_model:
            sentences = self.fast_run_cluster(sentences, ratio, algorithm, use_first)
        return " ".join(sentences)

    def combine_sentence(self,augment,main,degree=3):
        if DEBUG:
            print(augment,main)
        votes = set(main)
        last_sent = max(set(augment).union(set(main)))
        
        for n in augment:
            neighbours_positive = set([min(n+i,last_sent) for i in range(1,degree+1)])
            neighbours_neg = set([max(n-i,0) for i in range(1,degree+1)])
            neighbours = neighbours_positive.union(neighbours_neg)
            add = True
            if n not in votes:
                for neighbour in neighbours:
                    if neighbour in votes:
                        add = False
                        break
                if add:
                    votes.add(n) 

        return list(sorted(votes))

    def __call__(self,text,ratio,algorithm,use_first=True,min_length=None,max_length=None):
        return self.run(text,ratio,algorithm,use_first,min_length,max_length)

#%%
if __name__ == "__main__":
    import pickle
    a = Summarizer()
    filename = 'small_saved_model.pkl'
    pickle.dump(a, open(filename, 'wb'))

# %%
    with open('data/election_covid.txt') as fp:
        text = fp.read()
        a=Summarizer()
        z = a(text,ratio=0.2,algorithm='kmeans',use_first=True,min_length=10,max_length=500)
        print(z)

    # %%
    '''
    agglo chosen sentences = ([0, 14, 17, 18, 23, 28, 30, 37, 40, 49]
                        , [0, 1, 5, 8, 14, 17, 20, 26, 40, 47])

    kmeans chosen sentences = ([0, 14, 17, 18, 23, 28, 30, 37, 40, 49]
                        , [0, 1, 5, 8, 14, 17, 18, 20, 26, 47])
    '''
# %%
    with open('data/covid_malaysia.txt') as fp:
        text = """PUTRAJAYA: Malaysia will extend the movement control order (MCO) by two weeks until Apr 14 to contain the further spread of COVID-19, said Prime Minister Muhyiddin Yassin on Wednesday (Mar 25).
    He urged Malaysians to just stay at home to break the chain of infection.
    “The Health Ministry and the National Security Council have briefed me. The current trend is that new cases are still happening and will continue for a while until it stops,” he said in a televised address.
    Mr Muhyiddin added that it was the only way to contain the situation which seems to be getting worse by the day. He added that the announcement was made ahead so as to avoid public panic.
    "I know it is not easy to stay at home for a long time. I am sure there are many challenges. But the reality is that we have not faced something like this before and we would like to contain it as soon as possible.
    "The MCO thus far has helped in controlling the spread, but we cannot be too happy about it until we successfully have zero new cases," he said.
    The prime minister added that although the number of cases has plateaued, the Health Ministry expects the figure to increase further very soon.
    "So, we will keep making efforts to break the chain of infection. That said, I will have to announce this extension. I am sorry, but I am doing this for your well-being and your health," said Mr Muhyiddin.
    The Malaysian government had earlier imposed the MCO for two weeks from Mar 18 until Mar 31 to curb the spread of COVID-19.
    As part of the order, Malaysians are barred from travelling overseas while visitors are not allowed to enter the country. It also involves the closure of all government and private premises except for those providing essential services.
    All houses of worship and business premises are also closed except for supermarkets, grocery shops and convenience stores selling daily necessities.
    Despite the prime minister’s plea for all to stay at home, many Malaysians still continued to gather and socialise, and the army has to be deployed to the streets last weekend.
    The government has announced that offenders could be fined up to RM1,000 (US$225.70) or jailed up to six months, or both.
    As of Wednesday, Malaysia’s COVID-19 death toll was 19 and it reported more than 1,796 positive cases, making it the hardest-hit country in Southeast Asia."""
        z = a(text,ratio=0.2,algorithm='kmeans',use_first=True,min_length=10,max_length=500)
        print(z)
# %%

