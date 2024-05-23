# Importing necessary libraries
import pandas as pd # For data manipulation and reading CSV files
import streamlit as st # For creating the web app interface
from sklearn.feature_extraction.text import TfidfVectorizer # For text vectorization
from sklearn.metrics.pairwise import cosine_similarity # For calculating similarity between vectors
import numpy as np # For numerical operations

# Function to find the most similar question and get its answer
def GetSimilarQuestion(new_sentence, questions, tfidf_matrix, answers):
    # Transform the new sentence to a TF-IDF vector
    new_tfidf = vectorizer.transform([new_sentence])
    # Compute cosine similarity between the new TF-IDF vector and the existing TF-IDF matrix
    similarities = cosine_similarity(new_tfidf, tfidf_matrix)
    # Find the index of the most similar question
    most_similar_index = np.argmax(similarities)
    # Calculate the similarity percentage
    similarity_percentage = similarities[0, most_similar_index] * 100
    # Return the answer corresponding to the most similar question and the similarity percentage
    return answers[most_similar_index], similarity_percentage

# Function to provide the answer to the user's question
def AnswerTheQuestion(new_sentence, questions, tfidf_matrix, answers):
    # Get the most similar answer and similarity percentage
    most_similar_answer, similarity_percentage = GetSimilarQuestion(new_sentence, questions, tfidf_matrix, answers)
    # If similarity is above 70%, consider the answer confident enough
    if similarity_percentage > 70:
        response = {
            "answer": most_similar_answer,
            "confidence" : similarity_percentage
        }
    # Otherwise, indicate that the information is not available
    else:
        response = {
            "answer" : "Sorry, I don't have this information.",
            "confidence" : similarity_percentage
        }
    # Print the response (for debugging purposes)
    print(response)
    # Return the response
    return response

# Set the title of the Streamlit web app
st.title("Question Answering System")

# Create a file uploader widget to upload a CSV file
uploaded_file = st.file_uploader("Upload Your Question & Answers (CSV file)", type=["csv"])

# If a file is uploaded
if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file, encoding="utf-8")
    # Extract questions and answers from the DataFrame
    questions = df['question'].tolist()
    answers = df['answer'].tolist()

    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    # Fit and transform the questions to create a TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(questions)

    # Create a text input widget for the user to ask a question
    user_question = st.text_input("Ask your question here:")

    # Create a button to get the answer
    if st.button("Get Answer"):
        # If the user input is empty, show an error message
        if user_question.strip() == "":
            st.error("Please enter a valid question.")
        # Otherwise, process the question and get the answer
        else:
            response = AnswerTheQuestion(user_question, questions, tfidf_matrix, answers)
            # Display the answer
            st.write("Answer:", response["answer"])
