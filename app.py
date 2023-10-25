import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime
import tiktoken
from utils import pdf_to_text, generate_summary, save_vector_store, retrieval_augmented_generation


IMAGE = "https://miro.medium.com/v2/resize:fit:1400/1*02uoHJoYt3E7rylWEny02w.jpeg"
FEEDBACK_COLUMN_ONE = "Date"
FEEDBACK_COLUMN_TWO = "Summary"
FEEDBACK_COLUMN_THREE = "Feedback"
FEEDBACK_CSV = "feedback.csv"
PROMPT = """
Write a concise summary of the following:
{text}
CONCISE SUMMARY:
"""

#disable in session state
if 'disabled' not in st.session_state:
    st.session_state.disabled = False

if 'pdf' not in st.session_state:
    st.session_state.pdf = False
print(st.session_state)


# needful functions
def update_feedback(interact_date, user_summary, user_feedback):
    if not os.path.exists(FEEDBACK_CSV):
        feedback_data = pd.DataFrame(columns = [FEEDBACK_COLUMN_ONE, FEEDBACK_COLUMN_TWO, FEEDBACK_COLUMN_THREE])
        feedback_data.to_csv(FEEDBACK_CSV, index = False)

    feedback_df = pd.read_csv(FEEDBACK_CSV)

    metadata = {}
    metadata[FEEDBACK_COLUMN_ONE] = interact_date
    metadata[FEEDBACK_COLUMN_TWO] = user_summary
    metadata[FEEDBACK_COLUMN_THREE] = user_feedback
    feedback_df_new = pd.concat([feedback_df, pd.DataFrame(metadata, index = [0])], ignore_index = True)
    feedback_df_new.to_csv(FEEDBACK_CSV, index = False)
    print("Saved successfully")
    return True


def disabled():
    st.session_state.disabled = True


# variables
summary = None
saved_vector_store = None

# web design

# title
st.title('Climate Change Research Analyzer')

# image
st.image(IMAGE, caption="Climate Changes")

tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Question Answering", "Feedbacks", "Dashboard"])

with tab1:
    # upload a research
    st.subheader("Please Upload a Research Article in PDF Format")
    pdf = st.file_uploader("Upload PDF file",type="pdf")

    if pdf:
        with st.spinner("Extracting Texts and Getting the Summary..."):
                extracted_text = pdf_to_text(pdf)
                if extracted_text:
                    st.toast("Text Extraction Completed!!! 👍")
                summary = generate_summary(extracted_text, PROMPT)
                saved_vector_store = save_vector_store(extracted_text)
                if save_vector_store:
                    st.toast("Saved to the Vector Store! 👍")
                if extracted_text and summary:
                        st.toast("Summary Generation is completed! 👍")

                with st.expander("View Extracted Text"):
                        st.subheader("Extracted Text from the PDF")
                        st.write(extracted_text)

                # summary
                st.subheader("Generated Summary")
                st.write(summary)

                with st.sidebar:
                    st.subheader("Extracted Texts")
                    # display the extracted texts
                    with st.expander("View Extracted Text"):
                        st.subheader("Extracted Text from the PDF")
                        st.write(extracted_text)
                    # summary
                    st.subheader("Generated Summary")
                    st.write(summary)

with tab2:
    if not saved_vector_store:
        st.error("Please upload a PDF before this method 🚨")
    else:
        query = st.text_input("Please Enter Your Question")
        if query and saved_vector_store:
            with st.status("Getting response......"):
                st.write(retrieval_augmented_generation(query, saved_vector_store))

with tab3:
    st.header("Feedback Page")
    st.subheader("Please provide your feedback with web interaction")
    st.markdown("**Are you satidfied with the service?**")
    # feedback state
    feedback_state = False

    if not summary:
        st.error("Please generate a summary before giving a feedback 🚨")
    else:
        #create columns
        col1, col2 = st.columns(2)

        with col1:
            #set feedback buttons
            if st.button("Yes", type = "primary", on_click=disabled, disabled=st.session_state.disabled, use_container_width = True):
                feedback_state = update_feedback(datetime.now(), summary, "Yes")

        with col2:
            if st.button("No", type = "primary", on_click=disabled, disabled=st.session_state.disabled, use_container_width = True):
                feedback_state = update_feedback(datetime.now(), summary, "No")

        #say thank you if the feedback is given
        if feedback_state:
            st.header("Thank You For Your Feedback!")

with tab4:
    st.header("Feedback Analysis")
    st.subheader("Feedback Data")

    if not os.path.exists(FEEDBACK_CSV):
        st.error("Take Some Feedbacks from Users to view the Analytics 🚨")
    else:
        #read the data
        feedbacks = pd.read_csv(FEEDBACK_CSV)
        st.dataframe(feedbacks)
        #count plot
        sns.set_theme(style="whitegrid", palette="pastel")
        chart_fig = plt.figure(figsize=(5,5))
        sns.countplot(data = feedbacks, x = FEEDBACK_COLUMN_THREE)
        plt.xlabel("Feedback")
        plt.ylabel("Count")
        plt.title("Feedback Distribution")
        st.pyplot(chart_fig)
                    



