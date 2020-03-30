#%%
from spacy.lang.en import English


class SentenceHandler():

    def __init__(self, language = English):
        self.nlp = language()
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def process(self, body: str, min_length: int = 40, max_length: int = 600):
        body = body.replace('\n','')
        doc = self.nlp(body)
        return [c.string.strip() for c in doc.sents if max_length > len(c.string.strip()) > min_length]

    def __call__(self, body: str, min_length: int = 40, max_length: int = 600):
        return self.process(body, min_length, max_length)

# %%
if __name__ == "__main__":
    a = SentenceHandler()
    c = a("""The objective of this Standard is to establish the principles that an entity shall apply to
    report useful information to users of financial statements about the nature, amount,
    timing and uncertainty of revenue and cash flows arising from a contract with a
    customer.
    Meeting the objective
    2 To meet the objective in paragraph 1, the core principle of this Standard is that an entity shall
    recognise revenue to depict the transfer of promised goods or services to customers in an
    amount that reflects the consideration to which the entity expects to be entitled in exchange
    for those goods or services.
    3 An entity shall consider the terms of the contract and all relevant facts and circumstances
    when applying this Standard. An entity shall apply this Standard, including the use of any
    practical expedients, consistently to contracts with similar characteristics and in similar
    circumstances.
    4 This Standard specifies the accounting for an individual contract with a customer. However,
    as a practical expedient, an entity may apply this Standard to a portfolio of contracts (or
    performance obligations) with similar characteristics if the entity reasonably expects that the
    effects on the financial statements of applying this Standard to the portfolio would not differ
    materially from applying this Standard to the individual contracts (or performance obligations)
    within that portfolio. When accounting for a portfolio, an entity shall use estimates and
    assumptions that reflect the size and composition of the portfolio.
    """)
# %%

# %%
