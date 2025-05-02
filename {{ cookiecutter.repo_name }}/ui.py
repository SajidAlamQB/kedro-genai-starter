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
            response.raise_for_status()
            st.success("Pipeline triggered successfully!")

            results_response = requests.get("http://localhost:8000/hack/results")
            results_response.raise_for_status()
            output = results_response.json().get("content", "No content available.")
        except Exception as e:
            output = f"Error: {e}"
            st.error("Failed to fetch the response.")
else:
    output = ""

# Output display
st.markdown(output)
