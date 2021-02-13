import json
import random
import pandas as pd
import spacy
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split
from nltk.parse import CoreNLPParser
import re

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
           [clean_token(token) for token in answer_span.sent],\
           answer_span.start,\
           answer_span.end,\
           masked_answer_text_token,\
           masked_answer_sentence_token

def read_to_dataframe(filename, labeling, include_plausible_answers=False):
    nlp = spacy.load('en_core_web_sm')
    parser = CoreNLPParser(url='http://localhost:9000')

    df_paragraph = {"text_title":[],
                    "paragraph_text":[],
                    "paragraph_tokens":[],
                    "askable_tokens":[]}
    
    df_sentences = {"text_title":[],
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
                nlp_paragraph = nlp(paragraph_text)
                paragraph_tokens = [clean_token(t) for t in nlp_paragraph]
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
                                correct_answer_sentences.append(correct_answer_sentence)
                                correct_answer_token_index.append((answer_span_start, answer_span_end))
                                correct_masked_answer_texts_token.append(masked_answer_text_token)
                                correct_masked_answer_sentences_token.append(masked_answer_sentence_token)
                                correct_answer_sentences_token.append(correct_answer_sentence_token)
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
                                plausible_answer_sentences.append(plausible_answer_sentence)
                                plausible_answer_tokens_index.append((answer_span_start,answer_span_end))
                                plausible_masked_answer_texts_token.append(masked_answer_text_token)
                                plausible_masked_answer_sentences_token.append(masked_answer_sentence_token)
                                plausible_answer_sentences_token.append(plausible_answer_sentence_token)
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
                    df_sentences['text_title'].append(text_title)
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
    
    train_paragraph.to_csv("01_data/preprocessedData/train_paragraph.csv")
    dev_paragraph.to_csv("01_data/preprocessedData/dev_paragraph.csv")
    
    train_questions.to_csv("01_data/preprocessedData/train_questions.csv")
    dev_questions.to_csv("01_data/preprocessedData/dev_questions.csv")
    
    train_sentences.to_csv("01_data/preprocessedData/train_sentences.csv")
    dev_sentences.to_csv("01_data/preprocessedData/dev_sentences.csv")

    train_question_answer.to_csv("01_data/preprocessedData/train_question_answer_mapping.csv")
    dev_question_answer.to_csv("01_data/preprocessedData/dev_question_answer_mapping.csv")
    
    create_conll_file(train_sentences, "01_data/preprocessedData/train_conll.csv")
    create_conll_file(dev_sentences, "01_data/preprocessedData/dev_conll.csv")
    
    total_paragraph_df = pd.concat([train_paragraph, dev_paragraph])
    total_sentences_df = pd.concat([train_sentences, dev_sentences])
    total_question_df = pd.concat([train_questions, dev_questions])
    total_question_answer_df = pd.concat([train_question_answer, dev_question_answer])
    
    unique_text_titles = total_paragraph_df['text_title'].unique()

    train_dev_titles, test_titles = train_test_split(unique_text_titles, test_size=0.2, random_state=1)
    train_titles, dev_titles = train_test_split(train_dev_titles, test_size=0.25, random_state=1)

    random_train_paragraph_df = total_paragraph_df.loc[[title in train_titles for title in total_paragraph_df['text_title']]]
    random_dev_paragraph_df = total_paragraph_df.loc[[title in dev_titles for title in total_paragraph_df['text_title']]]
    random_test_paragraph_df = total_paragraph_df.loc[[title in test_titles for title in total_paragraph_df['text_title']]]

    random_train_sentences_df = total_sentences_df.loc[[title in train_titles for title in total_sentences_df['text_title']]]
    random_dev_sentences_df = total_sentences_df.loc[[title in dev_titles for title in total_sentences_df['text_title']]]
    random_test_sentences_df = total_sentences_df.loc[[title in test_titles for title in total_sentences_df['text_title']]]

    random_train_question_df = total_question_df.loc[[title in train_titles for title in total_question_df['text_title']]]
    random_dev_question_df = total_question_df.loc[[title in dev_titles for title in total_question_df['text_title']]]
    random_test_question_df = total_question_df.loc[[title in test_titles for title in total_question_df['text_title']]]

    random_train_question_answer_df = total_question_answer_df.loc[[title in train_titles for title in total_question_answer_df['text_title']]]
    random_dev_question_answer_df = total_question_answer_df.loc[[title in dev_titles for title in total_question_answer_df['text_title']]]
    random_test_question_answer_df = total_question_answer_df.loc[[title in test_titles for title in total_question_answer_df['text_title']]]

    random_train_paragraph_df.to_csv("01_data/preprocessedData/random_train_paragraph.csv")
    random_dev_paragraph_df.to_csv("01_data/preprocessedData/random_dev_paragraph.csv")
    random_test_paragraph_df.to_csv("01_data/preprocessedData/random_test_paragraph.csv")

    random_train_sentences_df.to_csv("01_data/preprocessedData/random_train_sentences.csv")
    random_dev_sentences_df.to_csv("01_data/preprocessedData/random_dev_sentences.csv")
    random_test_sentences_df.to_csv("01_data/preprocessedData/random_test_sentences.csv")

    random_train_question_df.to_csv("01_data/preprocessedData/random_train_question.csv")
    random_dev_question_df.to_csv("01_data/preprocessedData/random_dev_question.csv")
    random_test_question_df.to_csv("01_data/preprocessedData/random_test_question.csv")

    random_train_question_answer_df.to_csv("01_data/preprocessedData/random_train_question_answer.csv")
    random_dev_question_answer_df.to_csv("01_data/preprocessedData/random_dev_question_answer.csv")
    random_test_question_answer_df.to_csv("01_data/preprocessedData/random_test_question_answer.csv")

    create_conll_file(random_train_sentences_df, "01_data/preprocessedData/train_conll.csv")
    create_conll_file(random_dev_sentences_df, "01_data/preprocessedData/dev_conll.csv")
    create_conll_file(random_test_sentences_df, "01_data/preprocessedData/test_conll.csv")

    create_paragraph_conll_file(random_train_paragraph_df, "01_data/preprocessedData/train_paragraph_conll.csv")
    create_paragraph_conll_file(random_dev_paragraph_df, "01_data/preprocessedData/dev_paragraph_conll.csv")
    create_paragraph_conll_file(random_test_paragraph_df, "01_data/preprocessedData/test_paragraph_conll.csv")

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
        for index, row in df_sentences.iterrows():
            for token, tagging in zip(row["sentence_tokens"], row["askable_tokens"]):
                f.write("{}\t{}\n".format(token, tagging))
            f.write("-DOCSTART- -X- O O\n")

def create_paragraph_conll_file(df_paragraph, filename):
    with open(filename, "a") as f:
        for index, row in df_paragraph.iterrows():
            for token, tagging in zip(row["paragraph_tokens"], row["askable_tokens"]):
                f.write("{}\t{}\n".format(token, tagging))
            f.write("-DOCSTART- -X- O O\n")


def clean_token(token):
    token = token.text
    if token == "":
        return "<<EMPTY>>"
    else:
        token = re.sub("\n", "<<LINEBREAK>>",token)
        token = re.sub(r"\s", "<<WHITESPACE>>", token)
        return token
    

TRAIN_FILENAME = '01_data/quac/train_v0.2.json'
DEV_FILENAME = '01_data/quac/val_v0.2.json'
LABELING = "IO"

create_train_dev_test(TRAIN_FILENAME, DEV_FILENAME, LABELING)