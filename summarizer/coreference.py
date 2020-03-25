#%%
from spacy.lang.en import English
import neuralcoref
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sentence import SentenceHandler

DEBUG = False
def debug(*msges):
    if DEBUG:
        for msg in msges:
            print("******DEBUG::  "+msg)

class CoreferenceHandler(SentenceHandler):

    def __init__(self, language = "en_core_web_sm", greedyness: float = 0.45):
        super().__init__()
        import spacy
        self.nlp = spacy.load(language)
        neuralcoref.add_to_pipe(self.nlp)

    def process(self, body: str, min_length: int = 2, max_length: int = 600):
        coref_resolved_doc = self.nlp(body)._.coref_resolved
        doc = self.nlp(coref_resolved_doc)
        debug(body,coref_resolved_doc)
        return [c.string.strip() for c in doc.sents if max_length > len(c.string.strip()) > min_length]


a = CoreferenceHandler()
a.process(u'My sister has a dog. She loves him')

# %%
