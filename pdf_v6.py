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
import anthropic
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import copy
import multiprocessing

# Download NLTK data (stopwords and tokenizer)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

api_key = st.secrets["OPENAI_API_KEY"]

client2 = OpenAI(api_key= api_key)

tokenizer = GPT2Tokenizer.from_pretrained("./models/gpt2")

# Load a pre-trained sentence-transformers model for filtering
model = SentenceTransformer('./models/all-MiniLM-L6-v2')

def combine_paragraphs(paragraphs):
    combined_text = "\n".join(paragraphs)
    return combined_text

def extract_text_within_brackets(text):
    matches = re.findall(r'\[(.*?)\]', text, re.DOTALL)
    if matches:
        extracted_texts = [s.strip().strip('"') for s in matches[0].split('",')]
        return extracted_texts
    return []

def process_chunks(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
    return " ".join(filtered_words)

def preprocess_text_file(txt_file):
    with open(txt_file, "r", encoding="utf-8") as file:
        text = file.read()
    words = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
    return " ".join(filtered_words)

def preprocess_text(pdf_file):
    def extract_text_from_pdf(pdf_file2):
        doc = fitz.open(stream=pdf_file2.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    text = extract_text_from_pdf(pdf_file)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
    return " ".join(filtered_words)

def extra_pdf_chunks(client2_local,assistant_id):
    text = f"Extract all relevant paragraphs"
    try:
        thread = client2_local.beta.threads.create()
        thread_id = thread.id
        client2_local.beta.threads.messages.create(thread_id=thread_id, role="user", content=text)
        run = client2_local.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
            temperature=1
        )
        while True:
            run_status = client2_local.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            if run_status.status == "completed":
                break
            time.sleep(2)
        messages = client2_local.beta.threads.messages.list(thread_id=thread_id)
        return messages.data[0].content[0].text.value.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def extra_pdf_keywords(assistant_id, text):
    text = f"Analyse the text and extract the relevant keywords: {text}"
    try:
        thread = client2_local.beta.threads.create()
        thread_id = thread.id
        client2_local.beta.threads.messages.create(thread_id=thread_id, role="user", content=text)
        run = client2_local.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
            temperature=1
        )
        while True:
            run_status = client2_local.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            if run_status.status == "completed":
                break
            time.sleep(2)
        messages = client2_local.beta.threads.messages.list(thread_id=thread_id)
        return messages.data[0].content[0].text.value.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def assistant2(client2_local,vector_id, metric):
    instructions = f"""Extract all the chunks of text within the document in large paragraphs (512 seq length) that best answer the specified metric.:  
    Metric: {metric}
    OUTPUT FORMAT: Return results as a valid array of strings, with each paragraph as a separate element:
    "paragraph1", "paragraph2", "paragraph3", ...
    Think broadly about this topic to capture all relevant information, always return information"""
    assistant2 = client2_local.beta.assistants.create(
        name="Metric Extraction Assistant",
        instructions=instructions,
        temperature=2,
        top_p=0.33,
        model="gpt-4o",
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
    )
    return assistant2

def assistant(client2_local,vector_id, metric, description):
    instructions = f""" 
    ESG SPECIALIST KEYWORD EXTRACTION SYSTEM INSTRUCTION
    As an ESG specialist, analyze only the provided text to generate comprehensive keyword lists for data extraction of the specified metric along with its description, following these strict guidelines:
    Metric: {metric}
    Description: {description}
    1. EXTRACT ONLY WORDS THAT APPEAR VERBATIM in the provided text
    2. DO NOT include words from the metric ({metric}) and description ({description}) title itself in your extraction
    3. PRESENT RESULTS as a single array containing all keywords
    4. FOCUS EXCLUSIVELY on the provided content, with no external knowledge
    5. If none found in provided text then output "no matching keywords"
    LOOK FOR THESE CATEGORIES, BUT OUTPUT ALL FINDINGS IN A SINGLE ARRAY:
    * EXACT TERMINOLOGY (phrases and technical terms)
    * TECHNICAL ABBREVIATIONS AND NOTATION
    * SYNONYMS AND ALTERNATIVES
    * COMPONENT KEYWORDS
    * CONTEXTUAL INDICATORS (including headers)
    * LAYMAN TERMINOLOGY (simplified and non-technical terms)
    OUTPUT FORMAT: ["keyword1", "keyword2", "keyword3", "keyword4", "phrase1", "phrase2", "technical term1", "technical term2", "abbreviation1", "abbreviation2", "notation1", "notation2", "alternative phrase1", "alternative phrase2", "alternative term1", "alternative term2", "indicator1", "indicator2", "header1", "header2", "simplified term1", "simplified term2", "non-technical term1", "non-technical term2"]
    IMPORTANT VERIFICATION STEPS:
    * Ensure every single term appears verbatim in the original text
    * Remove any terms that are part of the metric title
    * Eliminate any terms not directly from the source text
    * Verify exact spelling and formatting matches the source text
    CRITICAL: Extract keywords ONLY from the text provided for analysis. Do not introduce words based on external knowledge or variations not present in the text. Remove ALL sources from the output.
    """
    assistant1 = client2_local.beta.assistants.create(
        name="Metric Extraction Assistant",
        instructions=instructions,
        model="gpt-4.5-preview-2025-02-27",
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
    )
    return assistant1

def count_tokens(text):
    return len(tokenizer.encode(text))

def send_metric(metric: str, unit: str, description: str, time_line, chunks, keywords, metric_id,pdf_name):
    url = "https://hook.eu2.make.com/uky6zgqw8frbvpejsmlqcv9ujsllvx7l"
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
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        print("Metric sent successfully!")
    else:
        print(f"Failed to send metric: {response.status_code}, {response.text}")

def extract_keywords_yake(sentence, num_keywords=10, n=3, dedup_lim=0.9, window_size=3):
    extractor = KeywordExtractor(
        n=n,
        dedupLim=dedup_lim,
        windowsSize=window_size,
        top=num_keywords
    )
    keywords = extractor.extract_keywords(sentence)
    return [kw[0] for kw in keywords[:num_keywords]]

def extract_relevant_pages(pdf_content, keywords, chunk_size=2, context_sentences=0):
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    relevant_pages = {}
    keywords = [kw.lower() for kw in keywords]
    for page_num in range(len(doc)):
        page = doc[page_num]
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
            chunks = [" ".join(relevant_sentences[i:i + chunk_size]) for i in range(0, len(relevant_sentences), chunk_size)]
            relevant_pages[page_num + 1] = chunks
    return relevant_pages

def filter_relevant_chunks(chunks, query, top_k=5, max_seq_length=250, return_indices=False):
    query_embedding = model.encode(query, convert_to_tensor=True)
    chunk_embeddings = []
    for chunk in chunks:
        chunk_tokens = model.tokenizer.encode(chunk)
        if len(chunk_tokens) > max_seq_length:
            segments = [chunk[i:i + max_seq_length] for i in range(0, len(chunk), max_seq_length)]
        else:
            segments = [chunk]
        segment_embeddings = []
        for segment in segments:
            segment_embedding = model.encode(segment, convert_to_tensor=True)
            segment_embeddings.append(segment_embedding)
        chunk_embedding = torch.mean(torch.stack(segment_embeddings), dim=0)
        chunk_embeddings.append(chunk_embedding)
    similarities = util.pytorch_cos_sim(query_embedding, torch.stack(chunk_embeddings))[0]
    top_indices = similarities.argsort(descending=True)[:top_k]
    if return_indices:
        return top_indices
    else:
        return [chunks[i] for i in top_indices]

def process_extracted_chunks(pages_with_keywords, query, max_tokens=5000, max_seq_length=250):
    all_chunks = []
    page_numbers = []
    for page_num, chunks in pages_with_keywords.items():
        all_chunks.extend(chunks)
        page_numbers.extend([page_num] * len(chunks))
    top_k = 10
    while top_k >= 5:
        relevant_indices = filter_relevant_chunks(all_chunks, query, top_k=top_k, max_seq_length=max_seq_length, return_indices=True)
        relevant_pages = {}
        for i in relevant_indices:
            page_num = page_numbers[i]
            if page_num not in relevant_pages:
                relevant_pages[page_num] = []
            relevant_pages[page_num].append(all_chunks[i])
        unique_relevant_pages = {}
        for page_num, chunks in relevant_pages.items():
            unique_relevant_pages[page_num] = list(set(chunks))
        combined_text = " ".join([chunk for chunks in unique_relevant_pages.values() for chunk in chunks])
        total_tokens = count_tokens(combined_text)
        st.write(f"Total tokens with top_k={top_k}: {total_tokens}")
        if total_tokens <= max_tokens:
            return unique_relevant_pages
        top_k -= 1
    if top_k < 5:
        relevant_indices = filter_relevant_chunks(all_chunks, query, top_k=5, max_seq_length=max_seq_length, return_indices=True)
        relevant_pages = {}
        for i in relevant_indices:
            page_num = page_numbers[i]
            if page_num not in relevant_pages:
                relevant_pages[page_num] = []
            relevant_pages[page_num].append(all_chunks[i])
        unique_relevant_pages = {}
        for page_num, chunks in relevant_pages.items():
            unique_relevant_pages[page_num] = list(set(chunks))
        combined_text = " ".join([chunk for chunks in unique_relevant_pages.values() for chunk in chunks])
        total_tokens = count_tokens(combined_text)
        if total_tokens > max_tokens:
            truncated_chunks = []
            total_tokens = 0
            for page_num, chunks in unique_relevant_pages.items():
                for chunk in chunks:
                    chunk_tokens = count_tokens(chunk)
                    if total_tokens + chunk_tokens <= max_tokens:
                        truncated_chunks.append((page_num, chunk))
                        total_tokens += chunk_tokens
                    else:
                        remaining_tokens = max_tokens - total_tokens
                        if remaining_tokens > 0:
                            truncated_chunk = tokenizer.decode(tokenizer.encode(chunk)[:remaining_tokens])
                            truncated_chunks.append((page_num, truncated_chunk))
                        break
            processed_chunks = {}
            for page_num, chunk in truncated_chunks:
                if page_num not in processed_chunks:
                    processed_chunks[page_num] = []
                processed_chunks[page_num].append(chunk)
            return processed_chunks
    return unique_relevant_pages


def process_metric(args):
    """
    Worker function for multiprocessing.
    Args:
        args: A tuple containing (index, row, new_vector_store_id, pdf_content, time_line).
    """
    print("im jhere")
    index, row, new_vector_store_id, pdf_content, time_line,pdf_name = args
    client2_local = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    try:
        # Convert pdf_content (bytes) back to BytesIO
        pdf_content = io.BytesIO(pdf_content)
        
        # Rest of the function logic
        metric = row['Metric']
        #description = row['Description']
        description = row.get('Description','')
        #metric_id = row['Id']
        metric_id = row.get('Id','')
        unit = row['Unit']

        st.write(f"Processing index: {index}, Metric: {metric}")

        vector_store_files = client2_local.vector_stores.files.list(
            vector_store_id=new_vector_store_id
        )
        print(f"Vector store files for index {index}: {vector_store_files}")

        chunks_assistant = assistant2(client2_local,new_vector_store_id, metric)
        print(f"Created chunks assistant for index {index}")

        chunks = extra_pdf_chunks(client2_local,chunks_assistant.id)
        print(f"Extracted chunks for index {index}")
        
        output_file_name = f"output2_{index}.txt"
        with open(output_file_name, "w", encoding="utf-8") as file:
            file.write(chunks)
        print(f"Saved chunks to output2.txt for index {index}")

        cleaned_text = preprocess_text_file(output_file_name)
        st.write(f"Preprocessed text for index {index}")

        keyword_assistant = assistant(client2_local,new_vector_store_id, metric, description)
        print(f"Created keyword assistant for index {index}")

        keywords = extra_pdf_keywords(client2_local,keyword_assistant.id, cleaned_text)
        st.write(f"Extracted keywords for index {index}")

        if pd.isna(description):
            description = row['Metric']
        if unit == "#":
            unit = '#'
        if unit == 'Text':
            unit = "Text"

        pages_with_keywords = extract_relevant_pages(pdf_content, keywords, chunk_size=5, context_sentences=2)
        st.write(f"Extracted relevant pages for index {index}")

        query = f"Given this Metric ({metric}) and unit ({unit}), what's the value over this timeline ({time_line})?"
        processed_chunks = process_extracted_chunks(pages_with_keywords, query, max_tokens=5000)
        print(f"Processed chunks for index {index}")

        dict_str = json.dumps(processed_chunks)
        print(f"Processed chunks as JSON for index {index}")

        if processed_chunks:
            send_metric(metric, unit, description, time_line, dict_str, keywords, metric_id,pdf_name)
            time.sleep(1)
        else:
            text = "No matching keywords found in the document."
            send_metric(metric, unit, description, time_line, text, keywords, metric_id,pdf_name)

        return index, dict_str, keywords
    except Exception as e:
        print(f"Error in process_metric for index {index}: {e}")
        return index, None, None

# Streamlit UI
st.title("📑 AI-Powered Data Extraction V2")
st.sidebar.header("Upload and Extract Data")
st.sidebar.header("Input Metric and Unit")
time_line = st.sidebar.text_input("Enter the timeline. E.g 2023")
uploaded_csv = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")
execute_button = st.sidebar.button("Execute")

if uploaded_file and execute_button and uploaded_csv:
    if uploaded_file is not None:
        # Read the PDF content once and store it in memory as bytes
        pdf_content = uploaded_file.read()  # This is a bytes object
        pdf_name = uploaded_file.name
        uploaded_file.seek(0)
        if not pdf_content:
            st.error("Uploaded file is empty. Please upload a valid PDF.")
        elif not pdf_content.startswith(b"%PDF"):
            st.error("Invalid PDF file. Please upload a proper PDF.")
        else:
            df = pd.read_csv(uploaded_csv)
            df = df
            stripped_text = preprocess_text(io.BytesIO(pdf_content))  # Create a BytesIO object for preprocessing
            with open("output.txt", "w", encoding="utf-8") as file:
                file.write(stripped_text)
            print("Preprocessed text saved to 'output.txt'.")
            vector_store = client2.vector_stores.create(name="Support FAQ")
            new_vector_store_id = vector_store.id
            try:
                response = client2.files.create(
                    file=open("output.txt", "rb"),
                    purpose="assistants"
                )
                st.write("File uploaded successfully!")
                print("File ID:", response.id)
                print("File Details:", response)
            except Exception as e:
                print("An error occurred while uploading the file:", e)
            client2.vector_stores.files.create(
                vector_store_id=new_vector_store_id,
                file_id=response.id
            )
            print("Before Processing")

            # Prepare arguments for multiprocessing
            args_list = [
                (index, row, new_vector_store_id, pdf_content, time_line,pdf_name)
                for index, row in df.iterrows()
            ]

            # Use multiprocessing.Pool
            print("hello")
            with st.spinner("Processing metrics..."):
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    results = list(executor.map(process_metric, args_list))

            # Display results in Streamlit
            all_success = True
            
            for index, dict_str, keywords in results:
                if dict_str is not None:
                    st.write(f"Processed: {index}")
                     
                    
                    #st.write(dict_str)
                    #st.write(keywords)
                else:
                    st.error(f"Failed to process index {index}")#
                    all_success = False
            
            if all_success:
                st.success("🎉 All metrics have been processed successfully!")
            else:
                st.warning("⚠️ Some metrics failed to process. Please review the errors above.")
