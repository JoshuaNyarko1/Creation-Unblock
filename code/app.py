import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import google.generativeai as genai


# Define functions for identifying genre and generating lyrics
def identify_genre_bpm_tempo(lyrics):
    # Prompt template for identifying genre, BPM, and tempo
    prompt_template = '''
    Analyze the given lyrics to identify the genre, BPM, and tempo of the song. Here are the provided lyrics:

    {lyrics}

    Please provide the predicted genre, BPM, and tempo in the format "Genre: XXX, BPM: XXX, Tempo: XXX".
    '''

    # Initialize the ChatGem model
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3,
                                   google_api_key="AIzaSyAItK4wztRexd8oVBJ7WvuA_el_neHn6QM")

    # Create prompt with input variables
    prompt = PromptTemplate(template=prompt_template, input_variables=["lyrics"])

    # Initialize LLMChain with model and prompt
    chain = LLMChain(llm=model, prompt=prompt)

    # Generate response
    response = chain({"lyrics": lyrics}, return_only_outputs=True)
    prediction = response['text'].strip()  # Extract prediction from response

    # Parse predicted values
    prediction_parts = prediction.split(",")
    predicted_genre = prediction_parts[0].split(":")[1].strip()
    predicted_bpm = prediction_parts[1].split(":")[1].strip()
    predicted_tempo = prediction_parts[2].split(":")[1].strip()

    return predicted_genre, predicted_bpm, predicted_tempo


# Define the function to generate lyrics
def generate_lyrics(genre, topic):
    # Prompt template for generating lyrics
    prompt_template = '''
    Act Like lyrics generator Generate song lyrics for a song with "{genre}" genre surrounding the topic of "{topic} Make Sure it should
    be of at least 2 minutes song". 
    '''

    # Initialize the ChatGem model
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3,
                                   google_api_key="AIzaSyAItK4wztRexd8oVBJ7WvuA_el_neHn6QM")

    # Create prompt with input variables
    prompt = PromptTemplate(template=prompt_template, input_variables=["genre", "topic"])

    # Initialize LLMChain with model and prompt
    chain = LLMChain(llm=model, prompt=prompt)

    # Generate response
    response = chain({"genre": genre, "topic": topic}, return_only_outputs=True)
    generated_lyrics = response['text'].strip()  # Extract generated lyrics from response

    return generated_lyrics


# Streamlit UI
def main():
    st.title("CreationUnblock")

    option = st.radio("Select an option:", ("Identify Genre, BPM, and Tempo", "Generate Lyrics"))

    if option == "Identify Genre, BPM, and Tempo":
        lyrics = st.text_area("Enter the lyrics:", height=200)
        if st.button("Identify Genre, BPM, and Tempo"):
            if lyrics:
                genre, bpm, tempo = identify_genre_bpm_tempo(lyrics)
                st.write("Predicted Genre:", genre)
                st.write("Predicted BPM:", bpm)
                st.write("Predicted Tempo:", tempo)
            else:
                st.warning("Please enter some lyrics.")

    elif option == "Generate Lyrics":
        genre = st.selectbox("Select Genre:", ("Pop", "Rock", "Hip Hop", "Country"))
        topic = st.text_input("Enter Topic of the Song:")
        if st.button("Generate Lyrics"):
            if topic:
                lyrics = generate_lyrics(genre, topic)
                st.write("Generated Lyrics:")
                st.write(lyrics)
            else:
                st.warning("Please enter the topic of the song.")


if __name__ == "__main__":
    main()