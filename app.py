import streamlit as st
import PyPDF2
from google import genai
from google.genai import types
import time
import json

# ==========================================
# 1. PDF EXTRACTION FUNCTION
# ==========================================
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip(): 
                text += page_text + "\n"
        
        if len(text.strip()) == 0:
            return "EMPTY_PDF"
        return text
    except Exception as e:
        return f"ERROR: {str(e)}"

# ==========================================
# 2. TEXT CHUNKING FUNCTION
# ==========================================
def chunk_text(text, max_chunk_size=10000):
    chunks = []
    for i in range(0, len(text), max_chunk_size):
        chunks.append(text[i:i+max_chunk_size])
    return chunks

# ==========================================
# 3. AI GENERATION FUNCTION
# ==========================================
def generate_ai_content(api_key, text_chunks, language, progress_bar=None):
    client = genai.Client(api_key=api_key)
    all_summaries, all_questions, errors = [], [], []
    total_chunks = len(text_chunks)

    for i, chunk in enumerate(text_chunks):
        prompt = f"""
        You are a helpful study assistant. Based on the following text in {language}, provide:
        1. A concise summary.
        2. Exactly 3 important questions.
        
        Format your response STRICTLY as a JSON object with two keys: "summary" and "questions".
        Example: {{"summary": "The text explains...", "questions": ["Q1?", "Q2?", "Q3?"]}}
        
        Text:
        {chunk}
        """
        
        retries = 3
        success = False
        while retries > 0 and not success:
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        response_mime_type="application/json"
                    )
                )
                result = json.loads(response.text)
                all_summaries.append(result.get("summary", ""))
                all_questions.extend(result.get("questions", []))
                success = True
            except Exception as e:
                retries -= 1
                if retries == 0:
                    errors.append(f"Part {i+1} process nahi ho paya: {str(e)}")
                else:
                    time.sleep(2 ** (3 - retries))

        if progress_bar:
            progress_bar.progress((i + 1) / total_chunks)

    final_summary = "\n\n".join(all_summaries)
    final_questions = "\n".join([f"{idx}. {q}" for idx, q in enumerate(all_questions, 1)])
    return final_summary, final_questions, errors

# ==========================================
# 4. MAIN UI (STREAMLIT APP)
# ==========================================
st.set_page_config(page_title="PDF Genius AI", page_icon="🤖", layout="wide")
st.title("📚 PDF Genius AI")
st.write("Upload your notes and let AI generate summaries and questions for the entire document!")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    summary_type = st.selectbox("Summary Language", ["English", "Hindi", "Hinglish"])

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if st.button("Generate Summary & Questions"):
    if not api_key:
        st.warning("Please enter your API Key in the sidebar.")
    elif not uploaded_file:
        st.warning("Please upload a PDF file.")
    else:
        with st.spinner("Extracting text from PDF..."):
            extracted_text = extract_text_from_pdf(uploaded_file)
        
        if extracted_text == "EMPTY_PDF":
            st.error("Is PDF mein koi selectable text nahi hai.")
        elif extracted_text.startswith("ERROR:"):
            st.error(f"PDF read error: {extracted_text}")
        else:
            text_chunks = chunk_text(extracted_text)
            st.info(f"PDF processed into {len(text_chunks)} chunks.")
            
            progress_bar = st.progress(0)
            with st.spinner(f"AI is analyzing in {summary_type}..."):
                final_summary, final_questions, errors = generate_ai_content(
                    api_key, text_chunks, summary_type, progress_bar
                )
            progress_bar.empty()
            
            if errors:
                st.warning(f"Kuch parts process nahi ho paye:\n" + "\n".join(errors))
            
            tab1, tab2 = st.tabs(["📝 Summary", "❓ Important Questions"])
            with tab1:
                st.markdown("### Document Summary")
                st.write(final_summary)
                st.download_button("📥 Download Summary", final_summary, "AI_Summary.txt", "text/plain")
            with tab2:
                st.markdown("### Important Questions")
                st.write(final_questions)
                st.download_button("📥 Download Questions", final_questions, "AI_Questions.txt", "text/plain")