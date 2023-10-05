import streamlit as st
import pandas as pd
from pyabsa import AspectPolarityClassification as APC
from pyabsa.tasks.AspectPolarityClassification import SentimentClassifier
import re


def main():
    st.set_page_config(
        page_title="ABSA Museum App",
        page_icon="🏛️",
        initial_sidebar_state="expanded",
        menu_items={
            'About': 'https://www.extremelycoolapp.com/help',
        }
    )
    st.title("Analisis Sentimen pada Aspek dalam Ulasan Museum")

    # Input text
    input_text = st.text_area("Input Text", "")
    input_aspects = st.text_input("Input Aspects", "")
    aspect_list = input_aspects.split(", ")

    # Perform NLP tasks
    if st.button("Analyze"):
        # Perform your NLP tasks here
        full_text = tag_aspect(input_text, aspect_list)
        # st.write(full_text)
        st.write(input_text)
        result_df = apc_test(full_text)
        st.dataframe(result_df, hide_index=True)

def tag_aspect(input_text, aspects):
    sentence = input_text
    for aspect in aspects:
        aspect_lower = aspect.lower()
        idx = re.search(aspect_lower, sentence.lower())
        if idx:
            begin_idx, end_idx = idx.span()
            sentence = sentence[:begin_idx] + "[B-ASP]" + sentence[begin_idx:end_idx] + "[E-ASP]" + sentence[end_idx:]
    return sentence

def apc_test(input_text):
    sentiment_classifier = APC.SentimentClassifier(checkpoint="C:\\Users\\malif\\PycharmProjects\\pythonProject2\\Model\\lowercase_revisi\\merge_1600")
    result_predict = []
    result = sentiment_classifier.predict(
        text=input_text,
        print_result=True,
        ignore_error=True,  # ignore an invalid example, if it is False, invalid examples will raise Exceptions
        eval_batch_size=32,
    )
    result_predict.append(result)
    result_df = pd.DataFrame(result_predict)
    result_df.drop(['text', 'ref_sentiment', 'ref_check','probs', 'perplexity'], axis=1, inplace=True)
    result_df = result_df.explode(['aspect', 'sentiment', 'confidence']).reset_index(drop=True)
    return result_df


if __name__ == "__main__":
    main()
