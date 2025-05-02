import streamlit as st
import requests

# Title
st.title("Kedro AMA")

# Input prompt
prompt = st.text_input("Enter your prompt:")

# Submit button
if st.button("Submit"):
    with st.spinner("Processing..."):
        try:
            response = requests.post(
                "http://localhost:8000/run/__default__",
                headers={"Content-Type": "application/json"},
                json={
                    "tags": ["agent_rag"],
                    "params": {"user_query": prompt}
                }
            )
            output = response.json().get("response", "No response received.")
            st.success("Response received successfully!")
        except Exception as e:
            output = f"Error: {e}"
            st.error("Failed to fetch the response.")
else:
    output = ""

# Output display
st.text_area("Output:", value=output, height=200)
