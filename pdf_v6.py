import streamlit as st
import re
import fitz  # PyMuPDF
import requests
import uuid
import json
import pandas as pd
import time
import io
from yake import KeywordExtractor
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2Tokenizer
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from openai import OpenAI
import concurrent.futures

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords',force = True)
nltk.download('punkt_tab')

# Initialize tokenizer and model once
tokenizer = GPT2Tokenizer.from_pretrained("./models/gpt2")
model = SentenceTransformer('./models/all-MiniLM-L6-v2')

def combine_paragraphs(paragraphs):
    return "\n".join(paragraphs)

def extract_text_within_brackets(text):
    matches = re.findall(r'\[(.*?)\]', text, re.DOTALL)
    return [s.strip().strip('"') for s in matches[0].split('",')] if matches else []

def preprocess_text_file(txt_file):
    with open(txt_file, "r", encoding="utf-8") as file:
        text = file.read()
    words = word_tokenize(text)
    try:
        stop_words = set(stopwords.words('english'))
    except AttributeError:
        nltk.download('stopwords', force=True)
        stop_words = set(stopwords.words('english'))
    return " ".join([word for word in words if word.lower() not in stop_words and word.isalpha()])

def preprocess_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "".join([page.get_text() for page in doc])
    words = word_tokenize(text)
    try:
        stop_words = set(stopwords.words('english'))
    except AttributeError:
        nltk.download('stopwords', force=True)
        stop_words = set(stopwords.words('english'))
    return " ".join([word for word in words if word.lower() not in stop_words and word.isalpha()])

def extra_pdf_chunks(client2_local, assistant_id):
    try:
        thread = client2_local.beta.threads.create()
        thread_id = thread.id
        client2_local.beta.threads.messages.create(thread_id=thread_id, role="user", content="Extract all relevant paragraphs")
        run = client2_local.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id, temperature=1)

        while True:
            run_status = client2_local.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            if run_status.status == "completed":
                break
            time.sleep(2)

        messages = client2_local.beta.threads.messages.list(thread_id=thread_id)
        return messages.data[0].content[0].text.value.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def extra_pdf_keywords(client2_local, assistant_id, text):
    try:
        thread = client2_local.beta.threads.create()
        thread_id = thread.id
        client2_local.beta.threads.messages.create(thread_id=thread_id, role="user", content=f"Analyse the text and extract the relevant keywords: {text}")
        run = client2_local.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id, temperature=1)

        while True:
            run_status = client2_local.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            if run_status.status == "completed":
                break
            time.sleep(2)

        messages = client2_local.beta.threads.messages.list(thread_id=thread_id)
        return messages.data[0].content[0].text.value.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def assistant2(client2_local, vector_id, metric):
    instructions = f"""Extract all the chunks related to: {metric}. Return a list of large paragraphs."""
    return client2_local.beta.assistants.create(
        name="Chunk Extractor",
        instructions=instructions,
        temperature=2,
        top_p=0.33,
        model="gpt-4.1",
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
    )

def assistant(client2_local, vector_id, metric, description):
    instructions = f"""
    Extract exact keywords only from provided text related to: {metric}.
    Ignore keywords already appearing in the metric/description.
    Output as a simple list.
    """
    return client2_local.beta.assistants.create(
        name="Keyword Extractor",
        instructions=instructions,
        model="gpt-4.1",
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
    )

def count_tokens(text):
    return len(tokenizer.encode(text))

def send_metric(metric, unit, description, time_line, chunks, keywords, metric_id, pdf_name):
    payload = {
        "id": str(uuid.uuid4()),
        "pdf name": pdf_name,
        "Metric Id": metric_id,
        "metric": metric,
        "unit": unit,
        "description": description,
        "Time line": time_line,
        "dict": chunks,
        "keywords": keywords
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post("https://hook.eu2.make.com/uky6zgqw8frbvpejsmlqcv9ujsllvx7l", json=payload, headers=headers)
    print("Metric sent" if response.status_code == 200 else f"Failed to send metric: {response.status_code}")

def extract_relevant_pages(pdf_content, keywords, chunk_size=2, context_sentences=0):
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    relevant_pages = {}
    keywords = [kw.lower() for kw in keywords]

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if any(kw in text.lower() for kw in keywords):
            sentences = re.split(r'(?<=[.!?])\s+', text)
            relevant_sentences = []
            for i, sentence in enumerate(sentences):
                if any(kw in sentence.lower() for kw in keywords):
                    start = max(0, i - context_sentences)
                    end = min(len(sentences), i + context_sentences + 1)
                    relevant_sentences.extend(sentences[start:end])
            relevant_sentences = list(dict.fromkeys(relevant_sentences))
            chunks = [" ".join(relevant_sentences[i:i+chunk_size]) for i in range(0, len(relevant_sentences), chunk_size)]
            relevant_pages[page_num] = chunks
    return relevant_pages

def filter_relevant_chunks(chunks, query, top_k=5, max_seq_length=250, return_indices=False):
    query_embedding = model.encode(query, convert_to_tensor=True)
    chunk_embeddings = [model.encode(chunk, convert_to_tensor=True) for chunk in chunks]
    similarities = util.pytorch_cos_sim(query_embedding, torch.stack(chunk_embeddings))[0]
    top_indices = similarities.argsort(descending=True)[:top_k]
    return top_indices if return_indices else [chunks[i] for i in top_indices]

def process_extracted_chunks(pages_with_keywords, query, max_tokens=5000, max_seq_length=250):
    all_chunks = [chunk for chunks in pages_with_keywords.values() for chunk in chunks]
    top_k = 10
    while top_k >= 5:
        relevant_chunks = filter_relevant_chunks(all_chunks, query, top_k, max_seq_length)
        combined_text = " ".join(relevant_chunks)
        if count_tokens(combined_text) <= max_tokens:
            return relevant_chunks
        top_k -= 1
    return []

def process_metric(args):
    index, row, pdf_content_bytes, time_line, pdf_name = args

    try:
        print(f"\n📌 [Index {index}] Starting process...")
        client2_local = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        pdf_content = io.BytesIO(pdf_content_bytes)

        # Step 1: Preprocess and upload
        print(f"🔍 [Index {index}] Preprocessing PDF...")
        stripped_text = preprocess_text(io.BytesIO(pdf_content_bytes))
        
        temp_text_filename = f"temp_text_{index}.txt"
        with open(temp_text_filename, "w", encoding="utf-8") as file:
            file.write(stripped_text)
        print(f"✅ [Index {index}] Text extracted and saved to {temp_text_filename}")

        print(f"📤 [Index {index}] Uploading file to OpenAI...")
        uploaded_file = client2_local.files.create(file=open(temp_text_filename, "rb"), purpose="assistants")
        vector_store = client2_local.vector_stores.create(name=f"MetricVectorStore_{index}")
        vector_store_id = vector_store.id
        client2_local.vector_stores.files.create(vector_store_id=vector_store_id, file_id=uploaded_file.id)
        print(f"✅ [Index {index}] File uploaded and linked to vector store {vector_store_id}")

        # Step 2: Assistants
        metric = row['Metric']
        unit = row['Unit']
        description = row.get('Description', '')
        metric_id = row.get('Id', '')

        print(f"🧠 [Index {index}] Creating assistant for chunks...")
        chunks_assistant = assistant2(client2_local, vector_store_id, metric)

        print(f"🔎 [Index {index}] Extracting chunks...")
        chunks = extra_pdf_chunks(client2_local, chunks_assistant.id)
        output_file_name = f"output2_{index}.txt"
        with open(output_file_name, "w", encoding="utf-8") as file:
            file.write(chunks)
        print(f"✅ [Index {index}] Chunks saved to {output_file_name}")

        print(f"🧹 [Index {index}] Cleaning extracted chunks...")
        cleaned_text = preprocess_text_file(output_file_name)

        print(f"🧠 [Index {index}] Creating assistant for keywords...")
        keyword_assistant = assistant(client2_local, vector_store_id, metric, description)

        print(f"🔍 [Index {index}] Extracting keywords...")
        keywords = extra_pdf_keywords(client2_local, keyword_assistant.id, cleaned_text)
        print(f"✅ [Index {index}] Keywords extracted: {keywords}")

        # Step 3: Process extracted pages
        print(f"📄 [Index {index}] Extracting relevant PDF pages...")
        pages_with_keywords = extract_relevant_pages(pdf_content, keywords, chunk_size=5, context_sentences=2)

        query = f"Metric: {metric}, Unit: {unit}, Timeline: {time_line}"
        print(f"🧠 [Index {index}] Filtering relevant chunks based on query: {query}")
        processed_chunks = process_extracted_chunks(pages_with_keywords, query, max_tokens=5000)

        dict_str = json.dumps(processed_chunks) if processed_chunks else "No matching chunks"
        print(f"📬 [Index {index}] Sending data to webhook...")
        send_metric(metric, unit, description, time_line, dict_str, keywords, metric_id, pdf_name)
        print(f"✅ [Index {index}] Completed!\n")

        return index, dict_str, keywords

    except Exception as e:
        print(f"❌ Error in process_metric for index {index}: {e}")
        return index, None, None

# --- Streamlit App ---
st.title("📑 Treety Data Extraction")
st.sidebar.header("Upload and Configure")

time_line = st.sidebar.text_input("Timeline (e.g., 2023)")
uploaded_csv = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
uploaded_pdf = st.sidebar.file_uploader("Upload PDF file", type=["pdf"])
execute_button = st.sidebar.button("Execute Extraction")

if uploaded_pdf and uploaded_csv and execute_button:
    pdf_content_bytes = uploaded_pdf.read()
    pdf_name = uploaded_pdf.name

    df = pd.read_csv(uploaded_csv)
    print("hello")
    st.write("Process has started...")

    args_list = [(index, row, pdf_content_bytes, time_line, pdf_name) for index, row in df.iterrows()]

    st.spinner("Processing your request...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(process_metric, args_list))

    all_success = all(dict_str is not None for index, dict_str, keywords in results)

    for index, dict_str, keywords in results:
        if dict_str:
            st.write(f"Processed index: {index}")
        else:
            st.error(f"Failed to process index: {index}")

    if all_success:
        st.success("🎉 All metrics processed successfully!")
    else:
        st.warning("⚠️ Some metrics failed, check logs.")

