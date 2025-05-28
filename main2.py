# ========================================
# 1. Import Required Libraries
# ========================================
import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline

# ========================================
# 2. Load the Pre-trained Summarization Model (cached)
# ========================================
@st.cache_resource
def load_model():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return pipeline("summarization", model=model, tokenizer=tokenizer, framework="pt")

summarizer = load_model()

# ========================================
# 3. Streamlit App Layout
# ========================================
st.title("📄 Text Summarization with Hugging Face")
st.write("Summarize long documents or articles using BART from Hugging Face 🤗.")

# === Option to Upload a Text File ===
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

# === Option to Paste Text ===
input_text = st.text_area("Or paste your text here", height=300)

# ========================================
# 4. Process Text Input
# ========================================
final_text = ""

if uploaded_file is not None:
    final_text = uploaded_file.read().decode("utf-8")
elif input_text:
    final_text = input_text

# ========================================
# 5. Generate Summary
# ========================================
if st.button("Summarize"):
    if not final_text.strip():
        st.warning("Please upload or enter some text first.")
    else:
        with st.spinner("Summarizing..."):
            summary = summarizer(final_text, max_length=150, min_length=30, do_sample=False)
        st.subheader("📝 Summary")
        st.success(summary[0]['summary_text'])
