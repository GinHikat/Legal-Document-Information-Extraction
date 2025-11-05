import sys, os
import pandas as pd
import json 
from collections import OrderedDict
import re
from underthesea import sent_tokenize

from final_re import RE
from final_ner import NER

from vncorenlp import VnCoreNLP
from PyPDF2 import PdfReader

with open('artifact/id2relation.json', 'r') as f:
    id2relation = json.load(f)
    
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

jar_path = os.path.join(BASE_DIR, "artifact", "VnCoreNLP", "VnCoreNLP-1.1.1.jar")

ner_annotator = VnCoreNLP(jar_path, annotators="wseg,pos,ner", max_heap_size='-Xmx2g')

ner = NER(
    model_path="artifact/model_bilstm_crf.pt",
    token2idx_path="artifact/token2idx.json",
    label2idx_path="artifact/label2idx.json",
    annotator = ner_annotator
)

re_model = RE(checkpoint = 'artifact/re_8_train_phobert_1_3.pth',
           use_phobert=True, id2relation=id2relation, encoder_layer=1, decoder_layer=3, use_rel_pos=False, freeze_train=True) #match the model configuration

check_mask = ['luật', 'thông', 'nghị', 'hiến', 'quyết', 'định', 'pháp', 'tư', 'điều', 'mục', 'phần', 'khoản']

def final_relation_check(text, df):
    re_result = re_model.predict(text)
    ner_result = ner.extract_document_metadata(text)

    # Safety check for Span
    if re_result is None or 'Span' not in re_result.columns or re_result['Span'].isna().all():
        return df

    span = str(re_result['Span'].iloc[0]).lower()
    span_tokens = re.findall(r'\w+', span)

    # Rule check if Span has check_mask then span is valid, else it's not relation
    if any(token in check_mask for token in span_tokens):
        meta = ner_result[['issue_date', 'title', 'document_id']].iloc[:1].reset_index(drop=True)
        rel = re_result.iloc[:1].reset_index(drop=True)
        combined = pd.concat([rel, meta], axis=1)
        df = pd.concat([df, combined], ignore_index=True)

    return df

def extract_sentences(text):
    sentences = []
    buffer = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        buffer.append(line)

        if line.endswith(';'):
            sent = ' '.join(buffer).strip()

            idx = sent.find("Căn cứ")
            if idx != -1:
                sent = sent[idx:].strip()

            sentences.append(sent)
            buffer = []

    return sentences

def relation_extraction(text):
    
    df = pd.DataFrame(columns = ['Text', 'Self Root', 'Relation', 'Span', 'issue_date', 'title', 'document_id'])
    
    first = sent_tokenize(text)[0]

    sents = extract_sentences(first)
    
    for sent in sents:
        try:
            df = final_relation_check(sent, df)
        except Exception:
            pass
    
    return df
        
if __name__ == "__main__":
    file_path = input("Enter the path to your file: ").strip()
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found.")
    else:
        # Check file type
        if file_path.lower().endswith(".pdf"):

            reader = PdfReader(file_path)
            content = ""
            for page in reader.pages:
                content += page.extract_text() or ""
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
        df_meta = ner.extract_document_metadata(content)
        df_relation = relation_extraction(content)
        
        print("/nSample File Content")
        print(sent_tokenize(content)[0])  
        
        print('=' * 50)
        print('Metadata')
        print('=' * 50)
        print(df_meta)
        
        print('=' * 50)
        print('Relation')
        print('=' * 50)
        print(df_relation)
