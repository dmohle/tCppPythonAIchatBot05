# langChainDemo05.py
# Nov 12, 2023
# dH, Fresno, CA

import chainlit as cl
import os
import openai
from constants import apikey
from langchain.chains import LLMChain  # Import LLMChain from langchain.chains
from langchain.prompts import PromptTemplate  # Import PromptTemplate from langchain.prompts
from langchain.llms import OpenAI  # Import OpenAI from langchain.llms

os.environ['OPENAI_API_KEY'] = apikey
openai.api_key = apikey

template = """ You are a Python programming teacher teaching the concept of 
{question} 
to smart high schools students.
Prepare a lesson plan for tomorrow's class, and include sample Python code with verbose comments.
 """

@cl.on_chat_start
def main():
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(
        prompt=prompt,
        llm=OpenAI(temperature=0.2, streaming=True),
        verbose=True
    )
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    llm_chain = cl.user_session.get("llm_chain")

    res = await llm_chain.acall(message.content, callbacks=[cl.AsyncLangchainCallbackHandler()])

    await cl.Message(content=res["text"]).send()
