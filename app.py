import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load Hugging Face Model
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-large"  # Replace with a model suitable for your use case
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Function to generate chatbot responses
def generate_response(user_input):
    prompt = f"Medical question: {user_input}\nResponse:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=200, num_return_sequences=1, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit Application
def main():
    st.title("Medical Assistant Chatbot 🤖")
    st.markdown(
        """
        **Welcome to the Medical Assistant Chatbot!**  
        Ask your medical-related questions below, and the chatbot will provide responses.  
        **Disclaimer:** This chatbot is for informational purposes only. Always consult a healthcare professional for medical advice.
        """
    )
    
    # User Input
    user_input = st.text_area("Enter your question:", key="user_input")
    
    if st.button("Get Response"):
        if user_input.strip() == "":
            st.error("Please enter a valid question.")
        else:
            with st.spinner("Generating response..."):
                response = generate_response(user_input)
            st.success(response)

if __name__ == "__main__":
    main()
