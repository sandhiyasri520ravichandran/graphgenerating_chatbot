import streamlit as st
from dotenv import load_dotenv
from langchain_experimental.agents import create_csv_agent  # Corrected import
from langchain.llms.base import LLM
from pydantic import BaseModel
import requests
import pandas as pd
import tempfile


class MistralLLM(LLM, BaseModel):  # Ensure Pydantic integration
    api_key: str
    endpoint: str = "https://api.mistral.ai/v1"
    model: str = "mistral-7b"  # Ensure you specify the model

    def _call(self, prompt: str, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Updated payload with messages field required by the Mistral API
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 100)
        }

        print(f"Calling Mistral API with model: {self.model}")  # Debugging line

        response = requests.post(f"{self.endpoint}/completions", json=payload, headers=headers)

        if response.status_code == 200:
            return response.json()["choices"][0]["text"]
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")

    def _llm_type(self) -> str:
        return self.model  # Ensure this returns a valid model type


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your CSV 📈")
    st.header("Ask your CSV 📈")

    user_csv = st.file_uploader("Upload a CSV file", type="csv")
    if user_csv is not None:
        # Read the CSV file directly from the uploaded file
        df = pd.read_csv(user_csv)
        st.write("Here's a preview of your CSV:", df.head())

        user_question = st.text_input("Ask a question about your csv")

        # Use your Mistral API key here
        mistral_api_key = "wm0O4ZbwMFFkEnQosUR3DIlTd9n3zfkm"
        llm = MistralLLM(api_key=mistral_api_key)  # Corrected instance creation

        # Save the uploaded CSV file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, mode="w", newline="") as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            tmp_file_path = tmp_file.name

        # Create CSV agent using Langchain
        agent_executor = create_csv_agent(
            llm,
            tmp_file_path,  # Pass the file path here, not the DataFrame
            verbose=True,
            allow_dangerous_code=True
        )

        if user_question and user_question != "":
            response = agent_executor.run(user_question)
            st.write(response)


if __name__ == "__main__":
    main()
