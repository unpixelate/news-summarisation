#%%
from transformers import *
import logging
import torch
import numpy as np
from numpy import ndarray
from typing import List

DEBUG = True
def debug(msg):
    if DEBUG:
        print(msg)

class BERT:

    """
    Base handler for BERT models.
    """

    base_model, base_tokenizer = BertModel,BertTokenizer

    def __init__(
        self,
        model: str='bert-base-uncased',
        custom_model: PreTrainedModel=None,
        custom_tokenizer: PreTrainedTokenizer=None
    ):
        if custom_model:
            self.model = custom_model
        else:
            self.model = BERT.base_model.from_pretrained('bert-base-uncased', output_hidden_states=True)

        if custom_tokenizer:
            self.tokenizer = custom_tokenizer
        else:
            self.tokenizer = BERT.base_tokenizer.from_pretrained('bert-base-uncased')

        self.model.eval()

    def tokenize_input(self, text: str) -> torch.tensor:
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return torch.tensor([indexed_tokens])

    def extract_embeddings(
        self,
        text: str,
        hidden: int=-2,
        squeeze: bool=False,
        reduce_option: str ='mean'
    ) -> ndarray:

        """
        Extracts the embeddings for the given text

        :param text: The text to extract embeddings for.
        :param hidden: The hidden layer to use for a readout handler
        :param squeeze: If we should squeeze the outputs (required for some layers)
        :param reduce_option: How we should reduce the items.
        :return: A numpy array.
        """

        tokens_tensor = self.tokenize_input(text)
        pooled, hidden_states = self.model(tokens_tensor)[-2:]

        if -1 > hidden > -12:

            if reduce_option == 'max':
                pooled = hidden_states[hidden].max(dim=1)[0]

            elif reduce_option == 'median':
                pooled = hidden_states[hidden].median(dim=1)[0]

            else:
                pooled = hidden_states[hidden].mean(dim=1)

        if squeeze:
            return pooled.detach().numpy().squeeze()

        return pooled

    def create_matrix(
        self,
        content: List[str],
        hidden: int=-2,
        reduce_option: str = 'mean'
    ) -> ndarray:

        return np.asarray([
            np.squeeze(self.extract_embeddings(t, hidden=hidden, reduce_option=reduce_option).data.numpy())
            for t in content
        ])

    def __call__(
        self,
        content: List[str],
        hidden: int= -2,
        reduce_option: str = 'mean'
    ) -> ndarray:
        return self.create_matrix(content, hidden, reduce_option)


# %%
BERT()(["hello", 'world' ,'this','is', 'matrix'])

# %%
