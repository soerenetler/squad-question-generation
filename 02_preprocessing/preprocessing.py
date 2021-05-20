import json
import random
import pandas as pd
import spacy
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split
from nltk.parse import CoreNLPParser
import re
import hashlib
import os
import glob
import argparse

def get_answer_sentence(nlp_paragraph, answer_start, answer_text):
    length = len(answer_text)

    while not any([answer_start == token.idx for token in nlp_paragraph]):
        #print(str(answer_start) + "-1")
        answer_start-=1

    while not any([answer_start+length == token.idx+len(token) for token in nlp_paragraph]):
        #print(str(length) + "+1")
        length += 1

    answer_span = nlp_paragraph.char_span(answer_start, answer_start+length)
    if answer_span.text != answer_text:
        print(answer_span.text + "<-" + answer_text)
    # TODO There should be a better way to handle this
    assert not answer_span is None

    #assert answer_span.text == answer_text, "answer_span.text != answer_text: answer span: {}, answer text: {}".format(answer_span.text, answer_text)
    
    masked_answer_text_token = []
    for token in nlp_paragraph:
        if token.i >= answer_span.start and token.i < answer_span.end:
            masked_answer_text_token.append("<<Answer>>")
        else:
            masked_answer_text_token.append(token.text.lower())

    masked_answer_sentence_token = []
    for token in answer_span.sent:
        if token.i >= answer_span.start and token.i < answer_span.end:
            masked_answer_sentence_token.append("<<Answer>>")
        else:
            masked_answer_sentence_token.append(token.text.lower())

    return answer_span.sent.text,\
           [clean(token.text) for token in answer_span.sent],\
           answer_span.start,\
           answer_span.end,\
           masked_answer_text_token,\
           masked_answer_sentence_token

def read_to_dataframe(filename, labeling, include_plausible_answers=False):
    spacy.require_gpu()
    nlp = spacy.load('en_core_web_sm')
    parser = CoreNLPParser(url='http://localhost:9000')

    df_paragraph = {"text_title":[],
                    "paragraph_text":[],
                    "paragraph_tokens":[],
                    "askable_tokens":[]}
    
    df_sentences = {"text_title":[],
                    "paragraph_id":[],
                    "sentence_text":[],
                    "sentence_tokens":[],
                    "askable_tokens":[],
                    "sentence_start":[],
                    "sentence_end":[]}

    df_question = {"text_title":[],
                    "paragraph_text":[],
                    "question_id":[],
                    "question_text":[],
                    "question_token":[],
                    'question_pos':[],
                    'question_tag':[],
                    'question_parse_tree':[],
                    'correct_answer_text':[],
                    'correct_answer_sentence':[],
                    'correct_answer_sentence_token':[],
                    #'correct_answer_sentences_parse_tree':[],
                    'correct_answer_char_index':[], 
                    'correct_answer_token_index':[],
                    'correct_masked_answer_text_token':[],
                    'correct_masked_answer_sentence_token':[], 

                    'plausible_answer_text':[],
                    'plausible_answer_sentence': [],
                    'plausible_answer_sentences_token': [],
                    'plausible_answer_char_index':[], 
                    'plausible_answer_token_index':[],
                    'plausible_masked_answer_text_token':[],
                    'plausible_masked_answer_sentence_token':[],
                    "paragraph_tokens":[]}

    with open(filename) as json_data:
        json_dict = json.load(json_data)
        for text in tqdm(json_dict['data']):
            text_title = text['title']
            for paragraph in text['paragraphs']:
                paragraph_text = paragraph['context']
                paragraph_id = hashlib.sha224(paragraph_text.encode('utf-8')).hexdigest()
                nlp_paragraph = nlp(paragraph_text)
                paragraph_tokens = [clean(t) for t in nlp_paragraph]
                askable_tokens = ["O"]*len(nlp_paragraph)
                for question in paragraph['qas']:
                    question_id = question["id"]
                    question_text = question["question"]
                    question_doc = nlp(question_text)
                    question_token = [token.text for token in question_doc]
                    question_pos = [token.pos_ for token in question_doc]
                    question_tag =[token.tag_ for token in question_doc]

                    question_parse_tree = ""#str(list(parser.parse(question_token))[0])

                    correct_answer_texts = []
                    correct_answer_char_index = []
                    correct_answer_token_index = []
                    correct_answer_sentences = []
                    correct_answer_sentences_token = []
                    #correct_answer_sentences_parse_tree = []
                    correct_masked_answer_texts_token = []
                    correct_masked_answer_sentences_token = []

                    plausible_answer_texts = []
                    plausible_answer_char_index = []
                    plausible_answer_tokens_index = []
                    plausible_answer_sentences = []
                    plausible_answer_sentences_token = []
                    plausible_masked_answer_texts_token = []
                    plausible_masked_answer_sentences_token = []

                    if question['answers']:
                        for answer in question['answers']:
                            correct_answer_texts.append(answer['text'])
                            correct_answer_char_index.append((answer['answer_start'], answer['answer_start'] + len(answer['text'])))

                            result = get_answer_sentence(nlp_paragraph,
                                                         answer['answer_start'],
                                                         answer['text'])
                            if not result is None:
                                correct_answer_sentence, correct_answer_sentence_token, answer_span_start, answer_span_end, masked_answer_text_token, masked_answer_sentence_token = result
                                correct_answer_sentences.append(clean(correct_answer_sentence))
                                correct_answer_token_index.append((answer_span_start, answer_span_end))
                                correct_masked_answer_texts_token.append(masked_answer_text_token)
                                correct_masked_answer_sentences_token.append(masked_answer_sentence_token)
                                correct_answer_sentences_token.append([clean(t) for t in correct_answer_sentence_token])
                                #correct_answer_sentences_parse_tree.append(str(list(parser.parse(correct_answer_sentence_token))[0]))
                                if labeling == "IOB":
                                    if (answer_span_end-answer_span_start) == 1:
                                        askable_tokens[answer_span_start] = "I-Answer"
                                    else:
                                        askable_tokens[answer_span_start] = "B-Answer"
                                        askable_tokens[answer_span_start+1:answer_span_end] = ["I-Answer"]*(answer_span_end-answer_span_start-1)
                                elif labeling == "IO":
                                    askable_tokens[answer_span_start:answer_span_end] = ["I-Answer"]*(answer_span_end-answer_span_start)
                                else:
                                    raise ValueError("Currently only IOB and IO labeling is supported")

                    if "plausible_answers" in question.keys() and question["plausible_answers"]:
                        for plausible_answer in question["plausible_answers"]:
                            plausible_answer_texts.append(plausible_answer['text'])
                            plausible_answer_char_index.append((plausible_answer['answer_start'], plausible_answer['answer_start'] + len(plausible_answer['text'])))

                            result = get_answer_sentence(nlp_paragraph,
                                                          plausible_answer['answer_start'],
                                                          plausible_answer['text'])
                            if not result is None:
                                plausible_answer_sentence, plausible_answer_sentence_token, answer_span_start, answer_span_end, masked_answer_text_token, masked_answer_sentence_token = result
                                plausible_answer_sentences.append(clean(plausible_answer_sentence))
                                plausible_answer_tokens_index.append((answer_span_start,answer_span_end))
                                plausible_masked_answer_texts_token.append(masked_answer_text_token)
                                plausible_masked_answer_sentences_token.append(masked_answer_sentence_token)
                                plausible_answer_sentences_token.append([clean(t) for t in plausible_answer_sentence_token])
                                if labeling == "IOB":
                                    if (answer_span_end-answer_span_start) == 1:
                                        askable_tokens[answer_span_start] = "I-Answer"
                                    else:
                                        askable_tokens[answer_span_start] = "B-Answer"
                                        askable_tokens[answer_span_start+1:answer_span_end] = ["I-Answer"]*(answer_span_end-answer_span_start-1)
                                elif labeling == "IO":
                                    askable_tokens[answer_span_start:answer_span_end] = ["I-Answer"]*(answer_span_end-answer_span_start)
                                else:
                                    raise ValueError("Currently only IOB and IO labeling is supported")
                
                    df_question['text_title'].append(text_title)
                    df_question['paragraph_text'].append(paragraph_text)
                    df_question['correct_answer_text'].append(correct_answer_texts)
                    df_question['correct_answer_sentence'].append(correct_answer_sentences)
                    df_question['correct_answer_sentence_token'].append(correct_answer_sentences_token)
                    #df_question['correct_answer_sentences_parse_tree'].append(correct_answer_sentences_parse_tree)
                    df_question['correct_answer_char_index'].append(correct_answer_char_index) 
                    df_question['correct_answer_token_index'].append(correct_answer_token_index)
                    df_question['correct_masked_answer_text_token'].append(correct_masked_answer_texts_token)
                    df_question['correct_masked_answer_sentence_token'].append(correct_masked_answer_sentences_token)
                    
                    df_question['plausible_answer_text'].append(plausible_answer_texts)
                    df_question['plausible_answer_sentence'].append(plausible_answer_sentences)
                    df_question['plausible_answer_sentences_token'].append(plausible_answer_sentences_token)
                    df_question['plausible_answer_char_index'].append(plausible_answer_char_index) 
                    df_question['plausible_answer_token_index'].append(plausible_answer_tokens_index)
                    df_question['plausible_masked_answer_text_token'].append(plausible_masked_answer_texts_token)
                    df_question['plausible_masked_answer_sentence_token'].append(plausible_masked_answer_sentences_token)
                    df_question['question_id'].append(question_id) 
                    df_question['question_text'].append(question_text)
                    df_question['question_token'].append(question_token)
                    df_question['question_pos'].append(question_pos)
                    df_question['question_tag'].append(question_tag)
                    df_question['question_parse_tree'].append(question_parse_tree)

                    df_question['paragraph_tokens'].append(paragraph_tokens) 

                df_paragraph['text_title'].append(text_title)
                df_paragraph['paragraph_text'].append(paragraph_text)
                df_paragraph['paragraph_tokens'].append(paragraph_tokens)
                df_paragraph['askable_tokens'].append(askable_tokens)
                                                                    

                for sentence in nlp_paragraph.doc.sents:
                    if not set(paragraph_tokens[sentence.start:sentence.end]).issubset(set(['<<WHITESPACE>>', 'CANNOTANSWER'])):
                        df_sentences['text_title'].append(text_title)
                        df_sentences['paragraph_id'].append(paragraph_id)
                        df_sentences['sentence_text'].append(re.sub("\n", "<<LINEBREAK>>", sentence.text))
                        df_sentences['sentence_tokens'].append(paragraph_tokens[sentence.start:sentence.end])
                        df_sentences['askable_tokens'].append(askable_tokens[sentence.start:sentence.end])
                        df_sentences['sentence_start'].append(sentence.start)
                        df_sentences['sentence_end'].append(sentence.end)

    return pd.DataFrame(data=df_paragraph), pd.DataFrame(data=df_sentences), pd.DataFrame(data=df_question)


def create_train_dev_test(train_filename, dev_filename, labeling):
    train_paragraph, train_sentences, train_questions = read_to_dataframe(train_filename, labeling, include_plausible_answers=True)
    train_question_answer = create_question_answer_mapping(train_questions)
    
    dev_paragraph, dev_sentences, dev_questions = read_to_dataframe(dev_filename, labeling, include_plausible_answers=True)
    dev_question_answer = create_question_answer_mapping(dev_questions)

    unique_text_titles = train_paragraph['text_title'].unique()
    train_titles, dev_titles = train_test_split(unique_text_titles, test_size=0.2, random_state=1)
    
    delete_files_in_folderfileList = glob.glob(data_folder + "01_data/preprocessedData/"+ dataset +"/paragraph/*")
    delete_files_in_folder(data_folder + "01_data/preprocessedData/"+ dataset +"/paragraph_conll/*")
    train_train_paragraph = train_paragraph.loc[[title in train_titles for title in train_paragraph['text_title']]]
    train_dev_paragraph = train_paragraph.loc[[title in dev_titles for title in train_paragraph['text_title']]]
    train_train_paragraph.to_csv(data_folder + "01_data/preprocessedData/"+ dataset +"/paragraph/train.csv")
    train_dev_paragraph.to_csv(data_folder + "01_data/preprocessedData/"+ dataset +"/paragraph/dev.csv")
    dev_paragraph.to_csv(data_folder + "01_data/preprocessedData/"+ dataset +"/paragraph/test.csv")
    create_paragraph_conll_file(train_train_paragraph, data_folder + "01_data/preprocessedData/"+ dataset +"/paragraph_conll/train.conll")
    create_paragraph_conll_file(train_dev_paragraph, data_folder + "01_data/preprocessedData/"+ dataset +"/paragraph_conll/dev.conll")
    create_paragraph_conll_file(dev_paragraph, data_folder + "01_data/preprocessedData/"+ dataset +"/paragraph_conll/test.conll")
    
    delete_files_in_folder(data_folder + "01_data/preprocessedData/"+ dataset +"/questions/*")
    train_train_questions = train_questions.loc[[title in train_titles for title in train_questions['text_title']]]
    train_dev_questions = train_questions.loc[[title in dev_titles for title in train_questions['text_title']]]
    train_train_questions.to_csv(data_folder + "01_data/preprocessedData/"+ dataset +"/questions/train.csv")
    train_dev_questions.to_csv(data_folder + "01_data/preprocessedData/"+ dataset +"/questions/dev.csv")
    dev_questions.to_csv(data_folder + "01_data/preprocessedData/"+ dataset +"/questions/test.csv")
    
    delete_files_in_folder(data_folder + "01_data/preprocessedData/"+ dataset +"/sentences/*")
    train_train_sentences = train_sentences.loc[[title in train_titles for title in train_sentences['text_title']]]
    train_dev_sentences = train_sentences.loc[[title in dev_titles for title in train_sentences['text_title']]]
    train_train_sentences.to_csv(data_folder + "01_data/preprocessedData/"+ dataset +"/sentences/train.csv")
    train_dev_sentences.to_csv(data_folder + "01_data/preprocessedData/"+ dataset +"/sentences/dev.csv")
    dev_sentences.to_csv(data_folder + "01_data/preprocessedData/"+ dataset +"/sentences/test.csv")

    delete_files_in_folder(data_folder + "01_data/preprocessedData/"+ dataset +"/question_answer/*")
    train_train_question_answer = train_question_answer.loc[[title in train_titles for title in train_question_answer['text_title']]]
    train_dev_question_answer = train_question_answer.loc[[title in dev_titles for title in train_question_answer['text_title']]]
    train_train_question_answer.to_csv(data_folder + "01_data/preprocessedData/"+ dataset +"/question_answer/train.csv")
    train_dev_question_answer.to_csv(data_folder + "01_data/preprocessedData/"+ dataset +"/question_answer/dev.csv")
    dev_question_answer.to_csv(data_folder + "01_data/preprocessedData/"+ dataset +"/question_answer/test.csv")
    
    delete_files_in_folder(data_folder + "01_data/preprocessedData/"+ dataset +"/sentences_conll/*")
    train_train_sentences = train_sentences.loc[[title in train_titles for title in train_sentences['text_title']]]
    train_dev_sentences = train_sentences.loc[[title in dev_titles for title in train_sentences['text_title']]]
    create_conll_file(train_train_sentences, data_folder + "01_data/preprocessedData/"+ dataset +"/sentences_conll/train.conll")
    create_conll_file(train_dev_sentences, data_folder + "01_data/preprocessedData/"+ dataset +"/sentences_conll/dev.conll")
    create_conll_file(dev_sentences, data_folder + "01_data/preprocessedData/"+ dataset +"/sentences_conll/test.conll")

    
def delete_files_in_folder(folder_path):
    fileList = glob.glob(folder_path)
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)

def create_question_answer_mapping(question_df):
    mapping_df = {
                  "text_title":[],
                  "question_text": [],
                  "question_token": [],
                  "question_pos": [],
                  "question_tag": [],
                  "question_parse_tree":[],
                  "answer_sentence": [],
                  "answer_sentence_token":[],
                  "answer_paragraph": [],
                  "masked_answer_paragraph": [],
                  "masked_answer_sentence": [],
                  "answer": [],
                  "unanswerable": []
                 }
    for index, row in question_df.iterrows():
        paragraph_text = row["paragraph_text"]
        text_title = row["text_title"]

        for answer_text, correct_answer_sentence, correct_answer_sentence_token, masked_answer_text, masked_answer_sentence in zip(row["correct_answer_text"], row["correct_answer_sentence"], row["correct_answer_sentence_token"], row["correct_masked_answer_text_token"], row["correct_masked_answer_sentence_token"]):
            if not set(correct_answer_sentence_token).issubset(set(['<<WHITESPACE>>', 'CANNOTANSWER'])):
                mapping_df["text_title"].append(text_title)
                mapping_df["question_text"].append(row["question_text"])
                mapping_df["question_token"].append(row["question_token"])
                mapping_df["question_pos"].append(row["question_pos"])
                mapping_df["question_tag"].append(row["question_tag"])
                mapping_df["question_parse_tree"].append(row["question_parse_tree"])
                mapping_df["answer_sentence"].append(correct_answer_sentence)
                mapping_df["answer_sentence_token"].append(correct_answer_sentence_token)
                mapping_df["answer_paragraph"].append(paragraph_text)
                mapping_df["masked_answer_paragraph"].append(masked_answer_text)
                mapping_df["masked_answer_sentence"].append(masked_answer_sentence)
                mapping_df["answer"].append(answer_text)
                mapping_df["unanswerable"].append(False)

    return pd.DataFrame(data=mapping_df)

def create_conll_file(df_sentences, filename):
    with open(filename, "a") as f:
        previous_paragraph_id = ""
        for index, row in df_sentences.iterrows():
            if row["paragraph_id"] != previous_paragraph_id:
                f.write("-DOCSTART- -X- O O\n")
            
            f.write("\n")
            previous_paragraph_id = row["paragraph_id"]
            for token, tagging in zip(row["sentence_tokens"], row["askable_tokens"]):
                f.write("{}\t{}\n".format(token, tagging))

def create_paragraph_conll_file(df_paragraph, filename):
    with open(filename, "a") as f:
        for index, row in df_paragraph.iterrows():
            for token, tagging in zip(row["paragraph_tokens"], row["askable_tokens"]):
                f.write("{}\t{}\n".format(token, tagging))
            f.write("-DOCSTART- -X- O O\n")


def clean(token):
    token.text
    if token == "":
        return "<<EMPTY>>"
    else:
        token = re.sub("\n", "<<LINEBREAK>>",token)
        token = re.sub(r"\s", "<<WHITESPACE>>", token)
        return token
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing')
    parser.add_argument('--data-folder',  dest='data', metavar='d', type=str, default="/content/gdrive/MyDrive/mt-qg-data/", 
                        help='number om samples')
    parser.add_argument('--dataset',  dest='dataset', metavar='a', type=str, default="squad", 
                        help='number om samples')

    args = parser.parse_args()

    data_folder = args.data
    dataset = args.dataset
    TRAIN_FILENAME =  data_folder + '01_data/rawData/'+ dataset +'/train.json'
    DEV_FILENAME = data_folder + '01_data/rawData/'+ dataset +'/dev.json'
    LABELING = "IO"

    create_train_dev_test(TRAIN_FILENAME, DEV_FILENAME, LABELING)