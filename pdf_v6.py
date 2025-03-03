from openai import OpenAI
import re
import time
import streamlit as st
import pandas as pd
import json

api_key = st.secrets["OPENAI_API_KEY"]



# OpenAI API Key
client = OpenAI(api_key=api_key)

st.title("📑 AI-Powered Data Extraction")
st.sidebar.header("Upload and Extract Data")

def assistant_5(vector_id, metric, unit, value, context):

    assistant5 = client.beta.assistants.create(
        name="Page Extraction Assistant",
        instructions=(
            f"""
        ## **Task**
        You are an AI assistant skilled in extracting precise information from documents. Given a metric, its value, unit, and context, your task is to identify the exact page number where this information appears in the document.

        ## **Input**
        - **Metric:** {metric}  
        - **Value:** {unit}  
        - **Unit:** {value}
        - **Context:**  {context}
        

        ## **Instructions**
        1. Search through the document and locate the exact page number where this information appears.  
        2. If multiple pages contain relevant information, list all page numbers.  
        3. Provide a **confidence score (0-100%)** for the accuracy of the page number retrieval. 
        """
        """
        ## Mandatory JSON Output Format

        ```json
        {
            "page_table": [
                {
                                     
                    "Page Number(s)": ""
                    "Confidence Score":""
                }
            ]
        }
        """
        ),
        model="gpt-4o",
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
    )

    return assistant5

def assistant_4(vector_id, metric, unit, value,timeline):
    assistant4 = client.beta.assistants.create(
        name="Metric Extraction Assistant",
        instructions=(
            f"""
        # AI Instructions for Extracting 1 Metric from a Document
        Search the document for the phrase {metric} along with the value {value} {unit} and the year {timeline}, and return the corresponding context.

        """
        """
        ## Mandatory JSON Output Format

        ```json
        {
            "context_table": [
                {
                                     
                    "Context": " "
                }
            ]
        }
        """
        ),
        model="gpt-4o-mini",
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
    )
    return assistant4

# ✅ Create Vector Store

    #st.success(f"✅ Vector Store Created with ID: {new_vector_store_id}")

def chatgpt_assistant(prompt, assistant_id):
    try:
        thread = client.beta.threads.create()
        thread_id = thread.id
        client.beta.threads.messages.create(thread_id=thread_id, role="user", content=prompt)
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
            temperature=0.01,  # Add temperature parameter
            top_p=1           # Add top_p parameter
        )

        #progress_bar = st.progress(0)
        #status_text = st.empty()

        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            if run_status.status == "completed":
                #progress_bar.progress(100)
                #status_text.success("✅ Extraction Complete!")
                break
            time.sleep(2)
            #progress_bar.progress(50)
            #status_text.text("⏳ Extracting data...")

        messages = client.beta.threads.messages.list(thread_id=thread_id)
        return messages.data[0].content[0].text.value.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def upload_file_to_openai(file_path):
    """
    Uploads a file to OpenAI and stores it in the specified vector store.
    Returns the file ID.
    """
    try:
        with st.spinner("Uploading file..."):
            file = client.files.create(
                file=open(file_path, "rb"),
                purpose="assistants"
            )
            file_id = file.id
           
            #st.success(f"✅ File Uploaded: {file_id}")
            return file_id
    except Exception as e:
        st.error(f"❌ Error Uploading File: {str(e)}")
        return None

def upload_file_vector(file_id):
    with st.spinner("Adding file to Vector Store..."):
        client.beta.vector_stores.files.create(
            vector_store_id=new_vector_store_id,
            file_id=file_id
    
        )
        #st.success(f"✅ File {file_id} Added to Vector Store {new_vector_store_id}")

def extract_json_from_output(output):
    """
    Extracts JSON content from a given output string.

    Args:
    - output (str): The string containing the JSON content within triple backticks.

    Returns:
    - dict: The parsed JSON content as a Python dictionary.
    """
    # Regular expression to find JSON content within triple backticks
    json_match = re.search(r"```json\s*(.*?)\s*```", output, re.DOTALL)

    if json_match:
        json_content = json_match.group(1).strip()
        try:
            # Parse the JSON content
            json_data = json.loads(json_content)
            return json_data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
    else:
        print("No JSON content found.")
        return None

# ✅ Define Assistants

def assistant_1(vector_id, metric, unit,timeline):

    assistant = client.beta.assistants.create(
        name="Data Extraction Assistant",
        instructions=(
            f"""**Objective:**
Extract {metric} specifically in the unit {unit}."


---

## 📌 Extraction Instructions:
1. **Extract the 1 Metric without exception:**
   - Ensure the {metric} is included in the output even if no value is found.
2. **Identify and Extract Values:**
   - Locate all instances of the metric and extract their corresponding numerical values.
3. **Recognize Units:**
   - Extract the unit mentioned alongside each value; if it differs from {unit}, record it as presented.
4. **Determine Time Period:**
   - Extract any mention of the relevant time frame, focusing on the year {timeline}.
5. **Handle Variations:**
   - Account for abbreviations, alternative units, and different formatting styles (e.g., decimals, percentages, or written numbers).

---
"""
""" ## 📊 Mandatory JSON Output Format:

```json
{
  "metrics_table": [
    {"Metric": "{metric}", "Value": "", "Unit": "{unit}", "Time Period": {timeline}}
  ]
}

"""
        ),
        model="gpt-4o-mini",
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vector_id]}},
    )

    return assistant



vector_store = client.beta.vector_stores.create(name="Support FAQ")
new_vector_store_id = vector_store.id


st.sidebar.header("Input Metric and Unit")
metric_unit = st.sidebar.file_uploader("Upload Metrics Unit", type="csv")
time_line = st.sidebar.text_input("Enter the timeline. E.g 2023")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")


execute_button = st.sidebar.button("Execute")

overall_rows = [] 



if execute_button and uploaded_file and metric_unit:
    if uploaded_file is not None:
        metric_df = pd.read_csv(metric_unit)
    # Save the uploaded file to a temporary location
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Upload the file to OpenAI
        file_id = upload_file_to_openai("temp.pdf")
        if file_id:
            upload_file_vector(file_id)

    progress_bar = st.progress(0)
    total_rows = len(metric_df)
    
    for index, row in metric_df.iterrows():
        
        print(row["Metric"])
        metric = f"**{str(row['Metric'])}**"
        unit = f"**{str(row["Unit"])}**"
        print(f"Metric: {metric}, Unit {unit}")
        user_prompt = f"Analyze {uploaded_file.name} and extract the {metric} value (in {unit}) over {time_line}."
        assistant = assistant_1(new_vector_store_id, metric, unit,time_line)

        json_data = None
        while json_data is None:
            response = chatgpt_assistant(user_prompt, assistant.id)
            json_data = extract_json_from_output(response)
            if json_data is None:
                print("Retrying... JSON data is None")
                print(response)
                time.sleep(2)  # Adding a short delay before retrying
        #st.write(json_data)

        value = json_data["metrics_table"][0]["Value"]
        metric = json_data["metrics_table"][0]["Metric"]
        unit = json_data["metrics_table"][0]["Unit"]

        if value != "":
            assistant2 = assistant_4(new_vector_store_id,metric,unit,value,time_line)

            user_prompt2 = f"Analyse the document, extract the context of {metric}, {value} and {unit}"

            json_data2 = None
            while json_data2 is None:
                response2 = chatgpt_assistant(user_prompt2, assistant2.id)
                json_data2 = extract_json_from_output(response2)
                if json_data2 is None:
                    print("Retrying JSON2... JSON data is None")
                    print(response2)
                    time.sleep(2)  # Adding a short delay before retrying
                #st.write(json_data2)

            context = json_data2["context_table"][0]["Context"]

            assistant3 = assistant_5(new_vector_store_id,metric,unit,value,context)
            user_prompt3 = f" Identify the exact page number where this information appears in the document:  {metric}, {value},{unit} and {context}"

            json_data3 = None
            while json_data3 is None:
                response3 = chatgpt_assistant(user_prompt3, assistant3.id)
                json_data3 = extract_json_from_output(response3)
                if json_data3 is None:
                    print("Retrying JSON3... JSON data is None")
                    print(response3)
                    time.sleep(2) 
            #st.write(json_data3)

            page = json_data3["page_table"][0]["Page Number(s)"]
            cf_score = json_data3["page_table"][0]["Confidence Score"]
            
        elif value == "":
            value = "Not Found"
            context = "N/A"
            page = "N/A"
            cf_score = "N/A"

        overall_rows.append({
                    "Metric": metric,
                    "Value": value,
                    "Unit": unit,
                    "Page Number(s)": page,
                    "Context": context,
                    "Confidence Score": cf_score
                })
        progress_bar.progress((index + 1) / total_rows)

        # Create a pandas DataFrame from the collected rows
    overall_df = pd.DataFrame(overall_rows)
    st.write("Overall DataFrame:")
    st.write(overall_df)
    #print(overall_df)




        


