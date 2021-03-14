import streamlit as st

import joblib
import eli5
from FeatureTransformer import FeatureTransformer
import pandas as pd
import ast
import spacy
from spacy import displacy
from Custom import custom_tokenizer
from sklearn_crfsuite import metrics

def print_annotated_text(token_list, y_preds, threshold=0.5, y_trues=None, features=None):
# title=\"{:3f}\", style=\"background-color:rgba(255, 0, 0, {:3f}){}\
# token_template.format(y_pred["I-Answer"], y_pred["I-Answer"], "",

    if y_trues is None:
        y_trues = ["O"] * len(y_preds)
    output = ""
    for token, y_pred, y_true, feature in zip(token_list, y_preds, y_trues, features):
        attributes = ""
        title = "; ".join([key[7:] + " " + str(value) for key, value in feature.items() if key.startswith("0:")])
        if y_true == "I-Answer":
            attributes += " text-decoration: underline;"
        if y_pred["I-Answer"] > threshold:
            attributes += " font-weight: bold;"  
        output += "<div class=\'tooltip2\' style=\'background-color:rgba(255, 0, 0, "+ str(y_pred["I-Answer"])+ ");" + attributes+"\'> " + token + "<div class=\"tooltiptext\">"+ title +"</div> </div>"
        output += """
        """
    return output

def print_text_4_title(df, model, text_title):
  tmp_df = df[df_test["text_title"] == text_title]

  y_true = tmp_df["askable_tokens"]

  nlp.tokenizer = custom_tokenizer(tmp_df, nlp)
  ner_nlp.tokenizer = custom_tokenizer(tmp_df, nlp)

  test_features = feature_transformer.transform(tmp_df["sentence_text"])

  y_pred_test = model.predict(test_features)
  y_pred_test_marginals = model.predict_proba(test_features)

  st.sidebar.markdown(metrics.flat_classification_report(y_true, y_pred_test, digits=5))


  st.write("""<style>
  .tooltip2 {
    position: relative;
    display: inline-block;
  }

  .tooltip2 .tooltiptext {
    visibility: hidden;
    width: 500px;
    background-color: #555;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 5px 0;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    margin-left: -250px;
    opacity: 0;
    transition: opacity 0.3s;
  }

  .tooltip2 .tooltiptext::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -5px;
    border-width: 5px;
    border-style: solid;
    border-color: #555 transparent transparent transparent;
  }

  .tooltip2:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
  }
  </style>
  """, unsafe_allow_html=True)

  for sentence_text, token_list, y_trues, y_preds, test_feature in zip(list(tmp_df["sentence_text"]), list(tmp_df["sentence_tokens"]), list(y_true), y_pred_test_marginals, test_features):
      st.write(print_annotated_text(token_list, y_preds, y_trues=y_trues, threshold=threshold, features=test_feature), unsafe_allow_html=True)
      
      HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
      html = displacy.render(ner_nlp(sentence_text), style="ent")
      # Newlines seem to mess with the rendering
      html = html.replace("\n", " ")
      st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)




nlp = spacy.load('en_core_web_sm')
ner_nlp = spacy.load('../../../spacy_target_answer/training/model-best')

classification_level = st.radio("level of classification: ", ("sentence", "paragraph"))

if classification_level=="sentence":
  model = joblib.load('../../best_estimator.pkl')
  TEST_FILENAME = "../../01_data/preprocessedData/random_test_sentences.csv"
  df_test = pd.read_csv(TEST_FILENAME)
  df_test["askable_tokens"] = [ast.literal_eval(t) for t in df_test["askable_tokens"]]
  df_test["sentence_tokens"] = [ast.literal_eval(t) for t in df_test["sentence_tokens"]]

  text_title = st.selectbox("Choose a text...", list(set(list(df_test["text_title"]))))
  threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5)

  feature_transformer = FeatureTransformer(nlp)

  print_text_4_title(df_test, model, text_title)
elif classification_level=="paragraph":
  st.write("Not supported yet!")










