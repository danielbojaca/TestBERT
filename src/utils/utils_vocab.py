import pickle
import json
import torch 
import numpy as np
import pandas as pd

from typing import (
    List, Iterable, Union, Tuple
)
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score

class Encoded :
    '''Class to store encoded tokens. Emulates HuggingFace's Encoding'''

    def __init__(self, tokens, ids):
        self.tokens = tokens
        self.ids = ids

    def extend(self, encoded):
        self.tokens += encoded.tokens
        self.ids += encoded.ids


class BasicTokenizer :
    '''Emulates a HuggingFace-like tokenizer'''

    def __init__(self, tokenizer, special_tokens:List[str]) -> None:
        self.tokenizer = tokenizer
        assert(isinstance(special_tokens, list))
        assert(np.all(isinstance(x, str) for x in special_tokens))
        self.special_tokens = special_tokens
        self.unknown_token = special_tokens[0]
        self.has_stoi = False

    def initialize_from_iterable(self, list_sentences:Iterable) -> None:
        assert isinstance(list_sentences, Iterable)
        self.itos = self.special_tokens
        self.stoi = {token:i for i, token in enumerate(self.special_tokens)}
        for sentence in list_sentences:
            tokens = self.tokenizer(sentence)
            for token in tokens:
                if token not in self.itos:
                    self.stoi[token] = len(self.itos)
                    self.itos.append(token)
        self.has_stoi = True

    def encode(self, text:Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        assert(self.has_stoi), f'Error: Run first initialize_from_iterable to initialize the tokenizer!'
        if isinstance(text, str):
            return self._encode_str(text)
        else:
            inicial = True
            for sentence in text:
                if inicial:
                    encoded = self._encode_str(sentence)
                    inicial = False
                else:
                    new_encoded = self._encode_str(sentence)
                    encoded.extend(new_encoded)
            return encoded

    def decode(self, list_ids:Union[List[int], List[List[int]]]) -> Union[List[str], List[List[str]]]:
        assert(self.has_stoi), f'Error: Run first initialize_from_iterable to initialize the tokenizer!'
        if np.all(isinstance(id, int) for id in list_ids):
            return self._decode_str(list_ids)
        else:
            list_tokens = list()
            for ids in list_ids:
                tokens = self._decode_str(ids)
                list_tokens.append(tokens)
            return list_tokens

    def _encode_str(self, sentence:str) -> List[int]:
        tokens = self.tokenizer(sentence)
        indices = [self.stoi.get(token, 0) for token in tokens]
        tokens = [self.unknown_token if id == 0 else tokens[i] for i, id in enumerate(indices)]
        encoded = Encoded(tokens, indices)
        return encoded
    
    def _decode_str(self, list_ids:str) -> List[int]:
        tokens = [self.itos[id] for id in list_ids]
        return tokens
    
    def get_vocab_size(self):
        assert(self.has_stoi), f'Error: Run first initialize_from_iterable to initialize the tokenizer!'
        return len(self.itos)

    def save(self, tokenizer_file):
        with open(tokenizer_file, 'wb') as f:  # open a text file
            pickle.dump(self.stoi, f) # serialize the list

    @staticmethod
    def create_using_stoi(tokenizer, special_tokens:List[str], tokenizer_file):
        with open(tokenizer_file, 'rb') as f:
            stoi = pickle.load(f) # deserialize using load()
        assert(isinstance(stoi, dict))
        me_tokenizer = BasicTokenizer(tokenizer, special_tokens)
        me_tokenizer.stoi = stoi
        me_tokenizer.itos = list(stoi.keys())
        me_tokenizer.has_stoi = True
        return me_tokenizer


# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]
    


class BERTDataset(Dataset):
    '''
    Bert dataset that assumes four columns: 
        - BERT Input, which consist of two sentences, starting with [CLS] and both ended by [SEP]
        - BERT Label, which contains the target masked token and all other tokens are [PAD]
        - Segment Label, which labels the tokens of the sentences: 1 for tokens in the first sentence and 2 for tokens in the second
        - Is Next, which labels whether the second sentence follows the first
    '''
    def __init__(self, df:pd.DataFrame):
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # print(row)
        try: 
            bert_input = torch.tensor(row['BERT Input'], dtype=torch.long)
            bert_label = torch.tensor(row['BERT Label'], dtype=torch.long)
            segment_label = torch.tensor(row['Segment Label'], dtype=torch.long)
            # segment_label = torch.tensor([int(x) for x in row['Segment Label'].split(',')], dtype=torch.long)
            is_next = torch.tensor(row['Is Next'], dtype=torch.long)
            # print('dentro de try')
            # print("BERT Input:", bert_input)
            # print("BERT Label:", bert_label)
            # print("BERT Segement Label:", segment_label)
            # print("BERT is next:", is_next)

        except Exception as e:
            # print('=>:', e)
            try:
                bert_input = torch.tensor(json.loads(row['BERT Input']), dtype=torch.long)
                bert_label = torch.tensor(json.loads(row['BERT Label']), dtype=torch.long)
                segment_label = torch.tensor(json.loads(row['Segment Label']), dtype=torch.long)
                # segment_label = torch.tensor([int(x) for x in row['Segment Label'].split(',')], dtype=torch.long)
                is_next = torch.tensor(row['Is Next'], dtype=torch.long)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for row {idx}: {e}")
                print("BERT Input:", row['BERT Input'])
                print("BERT Label:", row['BERT Label'])
                # Handle the error, e.g., by skipping this row or using default values
                return None  # or some default values
        
        return bert_input, bert_label, segment_label, is_next  # Include original_text if needed
    

class BERTDatasetNoLabels(Dataset):
    '''
    Bert dataset that assumes four columns: 
        - BERT Input, which consist of two sentences, starting with [CLS] and both ended by [SEP]
        - Segment Label, which labels the tokens of the sentences: 1 for tokens in the first sentence and 2 for tokens in the second
        - Is Next, which labels whether the second sentence follows the first
    '''
    def __init__(self, df:pd.DataFrame):
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        try:
            bert_input = torch.tensor(row['BERT Input'], dtype=torch.long)
            segment_label = torch.tensor(row['Segment Label'], dtype=torch.long)
            is_next = torch.tensor(row['Is Next'], dtype=torch.long)
        except Exception as e:
            print(e)
            try:
                bert_input = torch.tensor(json.loads(row['BERT Input']), dtype=torch.long)
                segment_label = torch.tensor(json.loads(row['Segment Label']), dtype=torch.long)
                # segment_label = torch.tensor([int(x) for x in row['Segment Label'].split(',')], dtype=torch.long)
                is_next = torch.tensor(row['Is Next'], dtype=torch.long)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for row {idx}: {e}")
                print("BERT Input:", row['BERT Input'])
                # Handle the error, e.g., by skipping this row or using default values
                return None  # or some default values
        
        return bert_input, segment_label, is_next  # Include original_text if needed



def evaluate(dataloader, model, loss_fn_mlm, loss_fn_nsp, device) -> Tuple[float, float, float]:
    '''Evaluate a BERT model'''

    model.eval()  # Turn off dropout and other training-specific behaviors

    total_loss = 0
    total_next_sentence_loss = 0
    total_mask_loss = 0
    total_batches = 0
    total_count = 0
    y_true, y_predict = list(), list()
    with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
        for bert_inputs, bert_labels, segment_labels, is_nexts in dataloader:
            # Forward pass
            next_sentence_prediction, masked_language = model(bert_inputs, segment_labels)

            # Calculate loss for next sentence prediction
            # Ensure is_nexts is of the correct shape for CrossEntropyLoss
            next_loss = loss_fn_nsp(next_sentence_prediction, is_nexts.view(-1))

            # Calculate loss for predicting masked tokens
            # Flatten both masked_language predictions and bert_labels to match CrossEntropyLoss input requirements
            mask_loss = loss_fn_mlm(masked_language.view(-1, masked_language.size(-1)), bert_labels.view(-1))

            # Sum up the two losses
            loss = next_loss + mask_loss
            if torch.isnan(loss):
                total_loss += 100_000
                # continue
            else:
                total_loss += loss.item()
                total_next_sentence_loss += next_loss.item()
                total_mask_loss += mask_loss.item()
                total_batches += 1

            # print('next_sentence_pred:', next_sentence_prediction)
            logits = torch.softmax(next_sentence_prediction, dim=1)
            # print('logits: ', logits)
            prediction = torch.argmax(logits, dim=1)         
            total_count += is_nexts.size(0)
            y_true.extend(is_nexts.view(-1).cpu().numpy().tolist())
            y_predict.extend(prediction.cpu().numpy().tolist())

    avg_loss = total_loss / (total_batches + 1)

    #print(f"Average Loss: {avg_loss:.4f}, Average Next Sentence Loss: {avg_next_sentence_loss:.4f}, Average Mask Loss: {avg_mask_loss:.4f}")
    acc = accuracy_score(y_true, y_predict)
    f1 = f1_score(y_true, y_predict)    
    # print(f'{y_predict=}')
    # print(f"Accuracy: {acc}")
    # print(f"F1 score: {f1}")
    return avg_loss, acc, f1