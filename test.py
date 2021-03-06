#%%
import json
import requests
url = 'http://127.0.0.1:5000'
text = """
PUTRAJAYA: Malaysia will extend the movement control order (MCO) by two weeks until Apr 14 to contain the further spread of COVID-19, said Prime Minister Muhyiddin Yassin on Wednesday (Mar 25).
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
As of Wednesday, Malaysia’s COVID-19 death toll was 19 and it reported more than 1,796 positive cases, making it the hardest-hit country in Southeast Asia.
"""
data = {'body': text}
data = json.dumps(data)
sx = requests.post(url,data)
print(sx.json())

# %%
