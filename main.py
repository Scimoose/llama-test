from langchain.llms import CTransformers
from langchain.chains import LLMChain
from langchain import PromptTemplate

import os
import streamlit as st


prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?"
    )

def load_llm():
    # Load the LLM model
    llm = CTransformers(
        model = "D:\Tools\LLMs\llama-2-7b-chat.ggmlv3.q6_K.bin",
        model_type = "llama",
        max_new_tokens = 512,
        temperature = 0.7
    )
    llm_chain = LLMChain(
        llm = llm,
        prompt = prompt, 
        )
    return llm_chain

def main():
    input_text = input("You: ")
    llm_chain = load_llm()
    output_text = llm_chain.run(input_text)
    print("Bot: " + output_text)


if __name__ == "__main__":
    main()