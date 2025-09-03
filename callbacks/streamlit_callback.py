from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, message_placeholder):
        self.message_placeholder = message_placeholder
        self.full_response = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.full_response += token
        self.message_placeholder.markdown(self.full_response + "▌")

    def on_llm_end(self, *args, **kwargs):
        self.message_placeholder.markdown(self.full_response)
