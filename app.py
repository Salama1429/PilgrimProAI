
import os
import json
from typing import Optional, Tuple
import requests

from google.cloud import texttospeech

import gradio as gr
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import langchain

from langchain.embeddings import OpenAIEmbeddings



from langchain.memory import ConversationBufferWindowMemory

from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

from langchain.vectorstores import FAISS

import replicate
import azure.cognitiveservices.speech as speechsdk
import openai

OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]
openai.api_key =os.environ["OPENAI_API_KEY"]
REPLICATE_API_TOKEN =os.environ["REPLICATE_API_TOKEN"]
subscription_azure = os.environ["subscription_azure"]
X_RapidAPI_Key=os.environ["X_RapidAPI_Key"]
os.environ['GOOGLE_APPLICATION_CREDENTIALS']='new-project-for-contest-a73a5a47fdd1.json'




embeddings = OpenAIEmbeddings()
db = FAISS.load_local("faiss_index", embeddings)

retriever = db.as_retriever(search_kwargs={"k": 12})



llm=ChatOpenAI(temperature=0, model_name="gpt-4-0613", verbose=True)

#memory
memory = ConversationBufferWindowMemory( k=2, return_messages=True, memory_key="chat_history", output_key='answer') #last 2

CONDENSE_QUESTION_PROMPT.template='''Given the following conversation and a follow up question, \
rephrase the follow up question to be a standalone question, \
and if there is no chat_history donot rephrase it just return it as it is, \
and donot rephrase it if it is not related to the chat_history. \
\n\n Chat History:\n<<<{chat_history}>>>\nFollow Up Input: <<<{question}>>> \n\n Standalone question in the same language of Follow Up Input:'''

question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)

prompt_template = """you are a helpful assistant, Use the following pieces of context to answer the question at the end. \
your answer should be short.

{context}

Question: <<<{question}>>>

short answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)


chain = ConversationalRetrievalChain(
    retriever=retriever,
    question_generator=question_generator,
    combine_docs_chain=doc_chain,
    memory=memory,
    get_chat_history=lambda h : h
)


#langchain function
def reply_red(question):
    result= chain(question)['answer']
    return [(question, result)], result



#2- stt function
def STT_Azure_(audio_path):
    audio_file= open(audio_path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]


#3- TTS function
def tts_azure_(i):
    #audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True) #it run the audio aloud
    audio_config = speechsdk.audio.AudioOutputConfig(filename="file.wav") # it saves the audio file

    # The language of the voice that speaks.
    speech_config.speech_synthesis_voice_name='en-us-guyneural'
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    text=i
    #speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()
    return speech_synthesizer.speak_text_async(text).get()




#4-idle video

html_video = '<video width="192" height="192" autoplay loop><source src="https://ugc-idle.s3-us-west-2.amazonaws.com/3698249e35991cadeaa5fddf3ba88c63.mp4" type="video/mp4" poster="photo_modaifar.jpg"></video>'
   


#5-make video using google tts
def make_video_(i):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=i)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # The response's audio_content is binary.
    with open("output.mp3", "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)

    output = replicate.run(
        "devxpy/cog-wav2lip:8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25eef",
        input={"face": open("photo_modaifar.jpg", "rb"),
              "audio": open("output.mp3", "rb")
              }
    )
    html_video = f'<video width="192" height="192" autoplay><source src={output} type="video/mp4" poster="photo_modaifar.jpg"></video>'

    return html_video  


#get hotels in city
def get_hotels_in_city(city_):
    
    #get city id
    url = "https://booking-com.p.rapidapi.com/v1/hotels/locations"    
    querystring = {"name":city_,"locale":"en-gb"}
    headers = {
    "X-RapidAPI-Key": X_RapidAPI_Key,
    "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    city_id_ =response.json()[2]['city_ufi']



    #get hotels by city id
    url = "https://booking-com.p.rapidapi.com/v1/static/hotels"
    querystring = {"page":"0","city_id":city_id_, "order_by":"popularity"}
    headers = {
    "X-RapidAPI-Key": X_RapidAPI_Key,
    "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    list_of_hotels=response.json()['result'][:4]

    new_list_of_hotels = []

    for item in list_of_hotels:
        new_item = {
            'name': item['name'],
            'hotel_id': item['hotel_id'],
            'url': item['url'],
            'hotel_class': item['hotel_class'],
            'address': item['address'],
            'hotel_description': item['hotel_description'],
            'latitude': item['latitude'],
            'longitude': item['longitude']
        }
        new_list_of_hotels.append(new_item)
    
    return new_list_of_hotels


function_descriptions = [
            {
                "name": "get_hotels_in_city",
                "description": "connect to api to get informations about avilable hotel in given city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city_": {
                            "type": "string",
                            "description": "the desired city, use this to get the available hotels in the same city",
                        },
                        
                    },
                    "required": ["city_"],
                },
            }
        ]


def run_conversation(user_input):
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": "You are best assistant ever!"},
            {"role": "user", "content": user_input}],
        functions= [
            {
                "name": "get_hotels_in_city",
                "description": "connect to api to get informations about avilable hotel in given city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city_": {
                            "type": "string",
                            "description": "the desired city, use this to get the available hotels in the same city",
                        },
                        
                    },
                    "required": ["city_"],
                },
            }
        ],
       
        function_call="auto",
    )
    message = response["choices"][0]["message"]

    #function calling
    if message.get("function_call"):
        function_name = message["function_call"]["name"]
        arguments = json.loads(message["function_call"]["arguments"])
        print(arguments)
        if function_name == "get_hotels_in_city":
            function_response = get_hotels_in_city(
                arguments.get("city_")
            )

        else:
            raise NotImplementedError()
        
        second_response = openai.ChatCompletion.create(
            model="gpt-4-0613",
             # get user input
             
            messages=[
                {"role": "user", "content": user_input},
                message,
                {
                    "role": "function",
                    "name": function_name,
                    "content": str(function_response),
                },
            ],
        )
        return second_response["choices"][0]["message"]["content"]
    else:
        return response


#get best flights

def get_best_flights(origin_city, destination_city, departureDate ):
    prompt_origin=f'what is the IATA code for {origin_city}, return IATA code only with no more words'
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[{"role": "system", "content": prompt_origin}] ,
        temperature=0)
    city_IATA_origin= response["choices"][0]["message"]["content"]
    
    prompt_destination=f'what is the IATA code for {destination_city}, return IATA code only with no more words'
    response = openai.ChatCompletion.create(
    model="gpt-4-0613",
    messages=[{"role": "system", "content": prompt_destination}] ,
    temperature=0)
    city_IATA_destination= response["choices"][0]["message"]["content"]
    
    url = "https://skyscanner44.p.rapidapi.com/search"
    querystring = {"adults":"1","origin":city_IATA_origin,"destination":city_IATA_destination,
                   "departureDate":departureDate,"currency":"USD","locale":"en-GB","market":"UK"}
    headers = {
        "X-RapidAPI-Key": X_RapidAPI_Key,
        "X-RapidAPI-Host": "skyscanner44.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    filghts_= response.json()["itineraries"]["buckets"] #is a list
    
    all_flights=[]
    for item in filghts_:

        details_flight={"name":item["name"],
                    "price":item["items"][0]["price"]["formatted"],
                    "id":item['items'][0]['legs'][0]['id'],
                    "origin":item['items'][0]['legs'][0]['origin']['name'],
                    "destination":item['items'][0]['legs'][0]['destination']['name'],
                    'durationInMinutes':item['items'][0]['legs'][0]['durationInMinutes'],
                    'departure':item['items'][0]['legs'][0]['departure'],
                    'arrival':item['items'][0]['legs'][0]['arrival'],
                    'carrier name':item['items'][0]['legs'][0]['carriers']['marketing'][0]['name'],
                    #'link':item["items"][0]['deeplink']
        
    }

        all_flights.append(details_flight)
        
    return all_flights


function_descriptions_flights = [
            {
                "name": "get_best_flights",
                "description": "connect to api to get informations about best flights",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin_city": {"type": "string","description": "the origin city of the flight for example riyadh"},
                        
                        "destination_city": {"type": "string","description": "the destination city of the flight for example cairo"},
                        "departureDate": {"type": "string","description": "the date of the flite for example 2023-10-11"},
                        
                        
                        
                    },
                    "required": ["origin_city", "destination_city", "departureDate"],
                },
            }
        ]


def run_conversation_flights(user_input):

    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": "You are best assistant ever!"},
            {"role": "user", "content": user_input}],
        functions= function_descriptions_flights,
        function_call="auto",
    )
    message = response["choices"][0]["message"]

    #function calling
    if message.get("function_call"):
        function_name = message["function_call"]["name"]
        arguments = json.loads(message["function_call"]["arguments"])
        print(arguments)
        if function_name == "get_best_flights":
            function_response = get_best_flights(
                origin_city=arguments.get("origin_city"),
                destination_city=arguments.get("destination_city"),
                departureDate=arguments.get("departureDate"),
            )

        else:
            raise NotImplementedError()
        
        second_response = openai.ChatCompletion.create(
            model="gpt-4-0613",
             # get user input
             
            messages=[
                {"role": "user", "content": user_input},
                message,
                {
                    "role": "function",
                    "name": function_name,
                    "content": str(function_response),
                },
            ],
        )
        return second_response["choices"][0]["message"]["content"]
    else:
        return response






#gradio app


title='''<html>
<head>
  <title>Image and Text Example</title>
  <style>
    .container {
      display: flex;
      align-items: center;
      justify-content: center;
      height: 30vh;
    }

    .image {
      flex: 1;
      text-align: center;
    }

    .text {
      flex: 1;
      text-align: left;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="image">
      <a href="https://ibb.co"><img src="https://i.ibb.co/s2ZM4Rz/2.png" alt="logo" border="0"></a>
    </div>
    <div class="text">
      <h1>Experience the future of Intelligent Virtual Human.</h1>
    </div>
  </div>
</body>
</html>'''


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML(title)

    with gr.Tab("Intelligent virtual Human"):
        with gr.Row():
            with gr.Column(scale=1, min_width=192, visible=True):
                video_html = gr.HTML(html_video)
                output_hidden_text=gr.Textbox(visible=False)
                output_hidden_text.change(make_video_, output_hidden_text, video_html)

            with gr.Column(scale=7):
                chatbot = gr.Chatbot(label='Chat')


        with gr.Row():
            message = gr.Textbox(label="Type your question about Hajj or Umrah",
                         value="What is the first step in hajj?",
                         lines=1, scale=7)
            submit = gr.Button(value="Send", variant="secondary", scale=1)

        with gr.Row():
            inputs_audio = gr.Microphone(source="microphone", type="filepath", label="Or Say It",
                                       interactive=True, streaming=False)
            inputs_audio.change(STT_Azure_, inputs=inputs_audio, outputs=message)  

        examples=gr.Examples(
        examples = [["""What is the first step in Hajj"""],

        ["What is the first step in Umrah"],
        ["what are the steps of Hajj"]
        ],
        inputs = message)    
            

        message.submit(reply_red, message, [chatbot,output_hidden_text])
        submit.click(reply_red, message, [chatbot,output_hidden_text])
    
        footer1='''
<p align="left" >
<strong>Notes: PilgrimPro AI is  designed as follows:</strong>
<br />
1- It has custom knowledge obtained from official Hajj Ministry guide books <a href='https://www.haj.gov.sa/Guides'>(Link🔗),</a> using langchain.
<br />
2- It has memory and can remember previous interactions.
<br />
3- It utilizes multiple AI technologies, including LLM (GPT-4), Speech-to-Text, Text-to-Speech, and Lip-Sync, enabling the creation of a talking virtual human.
</p>
'''   
        gr.HTML(footer1)
    
        
    with gr.Tab(label="Real Time Flights"):
        flights_textbox_output = gr.Textbox(label="The Answer",
             lines=1)
        with gr.Row():
            flights_textbox_input = gr.Textbox(label="ask about flights and make sure to include the origin city, destination city,and the date",
                 value="what are the best flights available from Riyadh to Jeddah on 9-9-2023",
                 lines=1, scale=7)        
            flights_submit_botton = gr.Button(value="Send", variant="secondary", scale=1)
            
        with gr.Row():
            inputs_audio_flights = gr.Microphone(source="microphone", type="filepath", label="Or Say It",
                                       interactive=True, streaming=False)
            inputs_audio_flights.change(STT_Azure_, inputs=inputs_audio_flights, outputs=flights_textbox_input) 
    
        examples_flights=gr.Examples(
        examples = [["what are the best flights available from Riyadh to Jeddah on 9-9-2023"],

        ["I want to travel from cairo to Jeddah on 11-10-2023 what are the best flights available"]
        ],
        inputs = flights_textbox_input)     
    
    
        flights_textbox_input.submit(run_conversation_flights, flights_textbox_input, flights_textbox_output)
        flights_submit_botton.click(run_conversation_flights, flights_textbox_input, flights_textbox_output)
        
        footer2='''
<p align="left" >
<strong>Notes: Real Time Flights is part of PilgrimPro AI and its designed as follows:</strong>
<br />
1- This service enables users to obtain updated information about flights in real-time.
<br />
2- Users can input natural language text without complications, and the AI (LLM 'GPT-4' & OpenAI Function calling) will handle the rest. 
<br />
3- It utilizes SkyScanner API <a href='https://www.skyscanner.com/flights'>(Link🔗)</a> to retrieve real-time flight information, leveraging OpenAI Function calling <a href='https://openai.com/blog/function-calling-and-other-api-updates'>(Link🔗)</a>.
</p>
'''   
        gr.HTML(footer2)
        
        
        
    with gr.Tab("Real Time Hotels"):
        hotel_textbox_output = gr.Textbox(label="The Answer",
             lines=1)
        with gr.Row():
            hotel_textbox_input = gr.Textbox(label="ask about hotels in specific city",
                 value="What are the best hotels in makkah?",
                 lines=1, scale=7)        
            hotel_submit_botton = gr.Button(value="Send", variant="secondary", scale=1)
            
        with gr.Row():
            inputs_audio_hotels = gr.Microphone(source="microphone", type="filepath", label="Or Say It",
                                       interactive=True, streaming=False)
            inputs_audio_hotels.change(STT_Azure_, inputs=inputs_audio_hotels, outputs=hotel_textbox_input) 
    
        examples_hotels=gr.Examples(
        examples = [["What are the best hotels in makkah?"],

        ["I want to know about the best hotels in Riyadh"]
        ],
        inputs = hotel_textbox_input)    
    
        hotel_textbox_input.submit(run_conversation, hotel_textbox_input, hotel_textbox_output)
        hotel_submit_botton.click(run_conversation, hotel_textbox_input, hotel_textbox_output)
        
        footer3='''
<p align="left" >
<strong>Notes: Real Time Hotels is part of PilgrimPro AI and its designed as follows:</strong>
<br />
1- This service enables users to get updated informations about Hotels in real-time.
<br />
2- Users can input natural language text without complications, and the AI (LLM 'GPT-4' & OpenAI Function calling) will handle the rest.
<br />
3- Its Using Booking.com API <a href='https://www.booking.com/'>(Link🔗)</a> to get the real-time Hotels information, utilizing the OpenAI Function calling <a href='https://openai.com/blog/function-calling-and-other-api-updates'>(Link🔗)</a>.
</p>
'''   
        gr.HTML(footer3)
        
    
    
demo.launch()







