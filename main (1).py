import os

import streamlit as st
from streamlit_option_menu import option_menu

from gemini_utility import load_gemini_pro_model, load_gemini_pro_vision_model, embedding_model_gemini, gemini_ask_me_anything

from PIL import Image

# Set up the Streamlit page
st.set_page_config(page_title='Gemini AI', page_icon='🧠', layout='wide')

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        'Gemini AI',
        ['Chatbot', 'Image Captioning', 'Embeded Text', 'Ask me anything'],
        menu_icon='robot',
        icons=['chat-dots-fill', 'card-image', 'body-text', 'question'],
        default_index=0)


# Translate roles for chat messages
def translate_role_for_streamlit(user_role):
    if user_role == 'model':
        return "assistant"
    else:
        return user_role


# Chatbot page
if selected == 'Chatbot':
    model = load_gemini_pro_model()

    # Create a chat session if it doesn't exist
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    # Display the title
    st.title("ChatBot")

    # Display the chat history
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    # Input field for user's message
    user_prompt = st.chat_input('Ask Gemini Pro')

    if user_prompt:
        st.chat_message('user').markdown(user_prompt)
        gemini_response = st.session_state.chat_session.send_message(
            user_prompt)

        # Display the Gemini AI response
        with st.chat_message('assistant'):
            st.markdown(gemini_response.text)

# Image Captioning Page

if selected == "Image Captioning":

    st.title("📷Image Captioning")

    # create a box for user to upload the image
    uploaded_image = st.file_uploader("Upload an image",
                                      type=["jpg", "jpeg", "png"])

    #creating button
    if st.button(
            "Generate Caption"
    ):  # Once this button is clicked the below code should get executed.
        img = Image.open(uploaded_image)
        col1, col2 = st.columns(2)
        with col1:
            resized_image = img.resize((800, 500))
            st.image(resized_image)

        # passing the image/prompt to my model
        default_prompt = "Write a caption for this image "
        # getting the response from gemini pro vision model
        caption = load_gemini_pro_vision_model(default_prompt, img)

        # Now displaying it
        with col2:
            st.info(caption)

# Now for text embedding
if selected == "Embeded Text":
    st.title("📝Embeded Text")

    # Giving a text box so the user can type
    input_text = st.text_area(label="",
                              placeholder="Enter the text to get embeddings")

    # Creating a button to get the embeddings
    if st.button("Get Embeddings"):
        response = embedding_model_gemini(input_text)
        st.markdown(response)

# Now model for ask me anything :-
if selected == "Ask me anything":
    st.title("🤔Ask me anything")

    # Giving a text box so the user can type
    input_text = st.text_area(label="", placeholder="Enter your text")

    # Now creating a button
    if st.button("Get response"):
        response = gemini_ask_me_anything(input_text)
        st.markdown(response)
