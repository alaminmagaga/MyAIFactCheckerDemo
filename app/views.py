# myapp/views.py
from django.shortcuts import render
from django.http import HttpResponse
from urllib.parse import urlparse
from .models import Factcheck
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import openai
import os
from translate import Translator
import requests
import speech_recognition as sr
import pyttsx3
from django.http import JsonResponse
from htmldate import find_date
from .authenticity_checker import check_news_authenticity, FactCheck, fact_check_statement
from tavily import TavilyClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
import csv
from django.contrib.auth.models import Group, User
from .models import UserReport
from rest_framework import permissions, viewsets
from .models import Factcheck
from retry import retry
from django.shortcuts import redirect, render
from .models import Factcheck
from requests.exceptions import ConnectionError
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage
# from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.agents import initialize_agent, AgentType
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.tools.tavily_search import TavilySearchResults
from .serializers import FactcheckSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status


# Set environment variables for OpenAI and Tavily API key
os.environ["OPENAI_API_KEY"] = ""
os.environ["TAVILY_API_KEY"] = ""

analyzer = SentimentIntensityAnalyzer()

translator = Translator(to_lang="ha")
french_translator = Translator(to_lang="fr")
igbo_translator = Translator(to_lang="ig")
yoruba_translator = Translator(to_lang="yo")
swahili_translator = Translator(to_lang="sw")
arabic_translator = Translator(to_lang="ar") 

def all_factchecks(request):
    # Fetch all Factcheck objects
    factchecks = Factcheck.objects.all()

    # Optionally, fetch all UserReport objects
    user_reports = UserReport.objects.all()

    # Pass the data to the template
    return render(request, 'all_factchecks.html', {
        'factchecks': factchecks,
        'user_reports': user_reports
    })


def export_factchecks_csv(request):
    # Create the HTTP response with CSV content type
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="factchecks.csv"'
    
    writer = csv.writer(response)
    writer.writerow([
        'ID', 'User Input News', 'Fact Check Result', 'Sentiment Label', 
        'Genuine URLs', 'Non-Authentic URLs', 'Number of Genuine Sources', 
        'Genuine URLs and Dates', 'Non-Authentic Sources'
    ])
    
    # Fetch all Factcheck objects
    factchecks = Factcheck.objects.all()
    
    for factcheck in factchecks:
        writer.writerow([
            factcheck.id,
            factcheck.user_input_news,
            factcheck.fresult,
            factcheck.sentiment_label,
            factcheck.genuine_urls,
            factcheck.non_authentic_urls,
            factcheck.num_genuine_sources,
            factcheck.genuine_urls_and_dates,
            factcheck.non_authentic_sources
        ])
    
    return response



def translate_text(text, translator):
    try:
        # Translate only the first element of the tuple
        translated_text = translator.translate(text[0]) if isinstance(text, tuple) else translator.translate(text)
        # If it's a tuple, create a new tuple with the translated text and the original label
        if isinstance(text, tuple):
            translated_result = (translated_text, text[1])
        else:
            translated_result = translated_text
        return translated_result
    except Exception as e:
        # Handle any exceptions that may occur during translation
        print(f"Translation error: {e}")
        return text  # Return the original text in case of an error

# Function to translate text to Yoruba
def translate_to_yoruba(text):
    try:
        translation = yoruba_translator.translate(text)
        # Check if the translation result is a tuple
        if isinstance(translation, tuple):
            # If it's a tuple, take the first element
            translation = translation[0]
        return translation
    except Exception as e:
        # Handle any exceptions that may occur during translation
        print(f"Translation error: {e}")
        return text
    
def translate_to_arabic(text):
    try:
        translation = arabic_translator.translate(text)
        # Check if the translation result is a tuple
        if isinstance(translation, tuple):
            # If it's a tuple, take the first element
            translation = translation[0]
        return translation
    except Exception as e:
        # Handle any exceptions that may occur during translation
        print(f"Translation error: {e}")
        return text
    
    
# Function to translate text to Swahili
def translate_to_swahili(text):
    try:
        translation = swahili_translator.translate(text)
        # Check if the translation result is a tuple
        if isinstance(translation, tuple):
            # If it's a tuple, take the first element
            translation = translation[0]
        return translation
    except Exception as e:
        # Handle any exceptions that may occur during translation
        print(f"Translation error: {e}")
        return text

    
# Function to translate text to Igbo
def translate_to_igbo(text):
    try:
        translation = igbo_translator.translate(text)
        # Check if the translation result is a tuple
        if isinstance(translation, tuple):
            # If it's a tuple, take the first element
            translation = translation[0]
        return translation
    except Exception as e:
        # Handle any exceptions that may occur during translation
        print(f"Translation error: {e}")
        return text
    
# Function to translate text to French
def translate_to_french(text):
    try:
        translation = french_translator.translate(text)
        # Check if the translation result is a tuple
        if isinstance(translation, tuple):
            # If it's a tuple, take the first element
            translation = translation[0]
        return translation
    except Exception as e:
        # Handle any exceptions that may occur during translation
        print(f"Translation error: {e}")
        return text


def index(request):
    return render(request, 'index.html')

def preview(request):
    return render(request,"preview.html")

def about(request):
    return render(request, 'about.html')

def fact(request):
    return render(request, 'factcheck.html')



# Hausa
def hausa(request):
    return render(request, 'hausa_index.html')
# Hausa
def hausa_factcheck(request):
    return render(request, 'hausa_factcheck.html')

def habout(request):
    return render(request, 'hausa_about.html')


def yoruba(request):
    return render(request, 'yoruba_index.html')

def yfactcheck(request):
    return render(request, 'yoruba_factcheck.html')

def yabout(request):
    return render(request, 'yoruba_about.html')



#Swahili
def swahili(request):
    return render(request, 'swahili_index.html')

def sfactcheck(request):
    return render(request, 'swahili_factcheck.html')

def sabout(request):
    return render(request, 'swahili_about.html')


def igbo(request):
    return render(request, 'igbo_index.html')

def ifactcheck(request):
    return render(request, 'igbo_factcheck.html')

def iabout(request):
    return render(request, 'igbo_about.html')




def french(request):
    return render(request, 'french_index.html')

def ffactcheck(request):
    return render(request, 'french_factcheck.html')

def fabout(request):
    return render(request, 'french_about.html')


def arabic(request):
    return render(request, 'arabic_index.html')

def arabout(request):
    return render(request, 'arabic_about.html')

def arafactcheck(request):
    return render(request, 'arabic_factcheck.html')

def araresult(request):
    return render(request, 'arabic_preview.html')



def speech_input(request):
    if request.method == 'POST':
        speech_input_result = request.POST.get('speechInputResult', '')
        return JsonResponse({'result': speech_input_result})

    return JsonResponse({'error': 'Invalid request method'})




messages = [{"role": "system", "content": "You are a Fact-Checking Assistant"}]

def CustomChatGPT(user_input):
    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply



# Initialize Tavily tool
tools = [TavilySearchResults(max_results=1)]

# # Initialize OpenAI chat 
# chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
chat = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    api_key="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" 
)

# Define chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful factchecking assistant. You may not need to use tools for every query - the user may just want to chat!",
        ),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Create the agent
agent = create_openai_tools_agent(chat, tools, prompt)

# Initialize the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def CustomChatTavily(user_input):
    messages.append({"role": "user", "content": user_input})
    
    # Invoke the agent with the user's question
    agent_response = agent_executor.invoke({"messages": [HumanMessage(content=user_input)]})
    
    # Extract the assistant's reply from the agent response
    if "output" in agent_response:
        tavily_reply = agent_response["output"]
    else:
        tavily_reply = "I'm sorry, I couldn't understand your question."
    
    messages.append({"role": "assistant", "content": tavily_reply})
    return tavily_reply


@csrf_exempt
def chat(request):
    user_input = request.POST.get('user_input', '')
    reply = CustomChatTavily(user_input)
    return JsonResponse({'assistant_reply': reply})


def report(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        message = request.POST.get('message')

        # Save the user report to the database
        UserReport.objects.create(name=name, message=message)

        # Optionally, you can add a success message to display on the same page
        success_message = "Report submitted successfully!"

        # Return a JSON response with the success message
        return JsonResponse({'success_message': success_message})

    # If it's not a POST request, you can still return an empty JSON response
    return JsonResponse({})

# def get_tavily_answer(query):
#     # Initialize the TavilyClient with your API key
#     tavily = TavilyClient(api_key="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")

#     # Use the TavilyClient to search for the answer to the user's question
#     result = tavily.qna_search(query=query, search_depth="advanced")

#     # Return the result
#     return result

def get_tavily_answer_with_retry(query):
    # Set up API key
    os.environ["TAVILY_API_KEY"] = "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"

    # Set up the agent
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
    search = TavilySearchAPIWrapper()
    tavily_tool = TavilySearchResults(api_wrapper=search)

    # Initialize the agent
    agent_chain = initialize_agent(
        [tavily_tool],
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # Run the agent with user input question
    output = agent_chain.run(query)
    
    return output
    


# def result(request):
#     if request.method == 'POST':
#         start_time = time.time()
#         # Check if the request contains speech input
#         speech_input_result = request.POST.get('speechInputResult', '')
#         hidden_input_value = request.POST.get('hiddenInput', '')
#         if speech_input_result:
#             # If speech input is provided, use it as user_query
#             user_query = speech_input_result
#             is_speech_input = True
#         else:
#             # If no speech input, use the text input as user_query
#             user_query = request.POST.get('query', '')
#             is_speech_input = False
        
#         tavily_answer = get_tavily_answer_with_retry(user_query)
#         authenticity_result, genuine_sources, num_genuine_sources, genuine_urls, non_authentic_urls, non_authentic_sources = check_news_authenticity(user_query)

#         genuine_urls_and_dates = {}
#         for url in genuine_urls:
#             try:
#                 html = requests.get(url).content.decode('utf-8')
#                 publication_date = find_date(html)
#                 genuine_urls_and_dates[url] = publication_date
#             except Exception as e:
#                 print(f"Error fetching date for {url}: {e}")
#                 genuine_urls_and_dates[url] = None


#         user_input_news = user_query

#         sentiment_scores = analyzer.polarity_scores(user_query)
#         sentiment_label = "Neutral"
#         if sentiment_scores['compound'] > 0.05:
#             sentiment_label = "Positive"
#         elif sentiment_scores['compound'] < -0.05:
#             sentiment_label = "Negative"

#         news_result = Factcheck.objects.create(
#             user_input_news=user_query,
#             fresult=tavily_answer,
#             sentiment_label=sentiment_label,
#             genuine_urls=genuine_urls,


            
         
#         )


#         return render(request, 'preview.html', { 'num_genuine_sources': num_genuine_sources,
#                                                  'genuine_urls': genuine_urls,
#                                                  'non_authentic_urls': non_authentic_urls,
#                                                  'n': non_authentic_sources,
#                                                  'user_input_news': user_input_news,
#                                                  'sentiment_label': sentiment_label,
#                                                  'sentiment_scores': sentiment_scores,
#                                                  'genuine_urls_and_dates': genuine_urls_and_dates,
#                                                  'tavily_answer': tavily_answer,
#                                                  'non_authentic_urls':non_authentic_urls,
#                                                  'hidden_input_value': hidden_input_value
#                                                 })

#     return render(request, 'factcheck.html')


from openai import OpenAI

client = OpenAI()

def result(request):
    if request.method == 'POST':
        start_time = time.time()
        # Check if the request contains speech input
        speech_input_result = request.POST.get('speechInputResult', '')
        hidden_input_value = request.POST.get('hiddenInput', '')
        if speech_input_result:
            # If speech input is provided, use it as user_query
            user_query = speech_input_result
            # Assuming 'OpenAI' is set up correctly elsewhere in your project
            is_speech_input = True
            
        else:
            # If no speech input, use the text input as user_query
            user_query = request.POST.get('query', '')
            is_speech_input = False
        
        tavily_answer = get_tavily_answer_with_retry(user_query)
        authenticity_result, genuine_sources, num_genuine_sources, genuine_urls, non_authentic_urls, non_authentic_sources,non_legit_length = check_news_authenticity(user_query)

        genuine_urls_and_dates = {}
        for url in genuine_urls:
            try:
                html = requests.get(url).content.decode('utf-8')
                publication_date = find_date(html)
                genuine_urls_and_dates[url] = publication_date
            except Exception as e:
                print(f"Error fetching date for {url}: {e}")
                genuine_urls_and_dates[url] = None

        user_input_news = user_query

        sentiment_scores = analyzer.polarity_scores(user_query)
        sentiment_label = "Neutral"
        if sentiment_scores['compound'] > 0.05:
            sentiment_label = "Positive"
        elif sentiment_scores['compound'] < -0.05:
            sentiment_label = "Negative"

        # Save factcheck result with primary key
        news_result = Factcheck.objects.create(
            user_input_news=user_query,
            fresult=tavily_answer,
            sentiment_label=sentiment_label,
            genuine_urls=genuine_urls,
            non_authentic_urls=non_authentic_urls,
            num_genuine_sources= num_genuine_sources,
            genuine_urls_and_dates=genuine_urls_and_dates,
            non_authentic_sources = non_legit_length


        )

        # Redirect to the result page with primary key
        return redirect('result_detail', slug=news_result.slug)

    return render(request, 'factcheck.html')

def result_detail(request, slug):
    # Retrieve Factcheck object using primary key
    factcheck = Factcheck.objects.get(slug=slug)
    hidden_input_value = request.POST.get('hiddenInput', '')

    return render(request, 'preview.html', {
        'factcheck': factcheck,
        'genuine_urls': factcheck.genuine_urls,
        'non_authentic_urls': factcheck.non_authentic_urls,
        'genuine_urls_and_dates': factcheck.genuine_urls_and_dates,
        'num_genuine_sources': factcheck.num_genuine_sources,
        'user_input_news': factcheck.user_input_news,
        'sentiment_label': factcheck.sentiment_label,
        'tavily_answer': factcheck.fresult,
        'n':factcheck.non_authentic_sources,
        'hidden_input_value': hidden_input_value,
        'created_at': factcheck.created_at,
        
    })

# def generate_text_to_speech(request):
#     client = OpenAI()
#     speech_input_result = request.POST.get('speechInputResult', '')
#     hidden_input_value = request.POST.get('hiddenInput', '')
#     if speech_input_result:
#         user_query = speech_input_result
#         response = client.audio.speech.create(
#                model="tts-1",
#                voice="alloy",
#                input=user_query,
#                )
    
#     # Save speech to file
#     response.stream_to_file("output.mp3")



def hausa_result(request):
    if request.method == 'POST':
        start_time = time.time()
        # Check if the request contains speech input
        speech_input_result = request.POST.get('speechInputResult', '')
        hidden_input_value = request.POST.get('hiddenInput', '')
        if speech_input_result:
            # If speech input is provided, use it as user_query
            user_query = speech_input_result
            is_speech_input = True
        else:
            # If no speech input, use the text input as user_query
            user_query = request.POST.get('query', '')
            is_speech_input = False
        
        tavily_answer = get_tavily_answer_with_retry(user_query)
        authenticity_result, genuine_sources, num_genuine_sources, genuine_urls, non_authentic_urls, non_authentic_sources,non_legit_length = check_news_authenticity(user_query)

        genuine_urls_and_dates = {}
        for url in genuine_urls:
            try:
                html = requests.get(url).content.decode('utf-8')
                publication_date = find_date(html)
                genuine_urls_and_dates[url] = publication_date
            except Exception as e:
                print(f"Error fetching date for {url}: {e}")
                genuine_urls_and_dates[url] = None


        user_input_news = user_query

        sentiment_scores = analyzer.polarity_scores(user_query)
        sentiment_label = "Neutral"
        if sentiment_scores['compound'] > 0.05:
            sentiment_label = "Positive"
        elif sentiment_scores['compound'] < -0.05:
            sentiment_label = "Negative"
   
        user_input_news = translate_text(user_query,translator)
        tavily_answer = translate_text(tavily_answer,translator)
        # Save factcheck result with primary key
        news_result = Factcheck.objects.create(
            user_input_news=user_query,
            fresult=tavily_answer,
            sentiment_label=sentiment_label,
            genuine_urls=genuine_urls,
            non_authentic_urls=non_authentic_urls,
            num_genuine_sources= num_genuine_sources,
            genuine_urls_and_dates=genuine_urls_and_dates,
            non_authentic_sources = non_legit_length

        )

        
        # Redirect to the result page with primary key
        return redirect('result_detail_hausa', slug=news_result.slug)
    return render(request, 'hausa_factcheck.html')

def result_detail_hausa(request, slug):
    # Retrieve Factcheck object using primary key
    factcheck = Factcheck.objects.get(slug=slug)

    return render(request, 'hausa_preview.html', {
        'factcheck': factcheck,
        'genuine_urls': factcheck.genuine_urls,
        'non_authentic_urls': factcheck.non_authentic_urls,
        'genuine_urls_and_dates': factcheck.genuine_urls_and_dates,
        'num_genuine_sources': factcheck.num_genuine_sources,
        'user_input_news': factcheck.user_input_news,
        'sentiment_label': factcheck.sentiment_label,
        'tavily_answer': factcheck.fresult,
        'n':factcheck.non_authentic_sources
        
    })



def yoruba_result(request):
    if request.method == 'POST':
        start_time = time.time()
        # Check if the request contains speech input
        speech_input_result = request.POST.get('speechInputResult', '')
        hidden_input_value = request.POST.get('hiddenInput', '')
        if speech_input_result:
            # If speech input is provided, use it as user_query
            user_query = speech_input_result
            is_speech_input = True
        else:
            # If no speech input, use the text input as user_query
            user_query = request.POST.get('query', '')
            is_speech_input = False
        
        tavily_answer = get_tavily_answer_with_retry(user_query)
        authenticity_result, genuine_sources, num_genuine_sources, genuine_urls, non_authentic_urls, non_authentic_sources, non_legit_length= check_news_authenticity(user_query)

        genuine_urls_and_dates = {}
        for url in genuine_urls:
            try:
                html = requests.get(url).content.decode('utf-8')
                publication_date = find_date(html)
                genuine_urls_and_dates[url] = publication_date
            except Exception as e:
                print(f"Error fetching date for {url}: {e}")
                genuine_urls_and_dates[url] = None


        user_input_news = user_query

        sentiment_scores = analyzer.polarity_scores(user_query)
        sentiment_label = "Neutral"
        if sentiment_scores['compound'] > 0.05:
            sentiment_label = "Positive"
        elif sentiment_scores['compound'] < -0.05:
            sentiment_label = "Negative"

        user_input_news = translate_to_yoruba(user_query)
        tavily_answer = translate_to_yoruba(tavily_answer)

        news_result = Factcheck.objects.create(
            user_input_news=user_query,
            fresult=tavily_answer,
            sentiment_label=sentiment_label,
            genuine_urls=genuine_urls,
            non_authentic_urls=non_authentic_urls,
            num_genuine_sources= num_genuine_sources,
            genuine_urls_and_dates=genuine_urls_and_dates,
            non_authentic_sources = non_legit_length

        )
        # Redirect to the result page with primary key
        return redirect('result_detail_yoruba', slug=news_result.slug)
    return render(request, 'yoruba_factcheck.html')

def result_detail_yoruba(request, slug):
    # Retrieve Factcheck object using primary key
    factcheck = Factcheck.objects.get(slug=slug)

    return render(request, 'yoruba_preview.html', {
        'factcheck': factcheck,
        'genuine_urls': factcheck.genuine_urls,
        'non_authentic_urls': factcheck.non_authentic_urls,
        'genuine_urls_and_dates': factcheck.genuine_urls_and_dates,
        'num_genuine_sources': factcheck.num_genuine_sources,
        'user_input_news': factcheck.user_input_news,
        'sentiment_label': factcheck.sentiment_label,
        'tavily_answer': factcheck.fresult,
        'n':factcheck.non_authentic_sources
        
    })

def igbo_result(request):
    if request.method == 'POST':
        start_time = time.time()
        # Check if the request contains speech input
        speech_input_result = request.POST.get('speechInputResult', '')
        hidden_input_value = request.POST.get('hiddenInput', '')
        if speech_input_result:
            # If speech input is provided, use it as user_query
            user_query = speech_input_result
            is_speech_input = True
        else:
            # If no speech input, use the text input as user_query
            user_query = request.POST.get('query', '')
            is_speech_input = False
        
        tavily_answer = get_tavily_answer_with_retry(user_query)
        authenticity_result, genuine_sources, num_genuine_sources, genuine_urls, non_authentic_urls, non_authentic_sources,non_legit_length = check_news_authenticity(user_query)

        genuine_urls_and_dates = {}
        for url in genuine_urls:
            try:
                html = requests.get(url).content.decode('utf-8')
                publication_date = find_date(html)
                genuine_urls_and_dates[url] = publication_date
            except Exception as e:
                print(f"Error fetching date for {url}: {e}")
                genuine_urls_and_dates[url] = None


        user_input_news = user_query

        sentiment_scores = analyzer.polarity_scores(user_query)
        sentiment_label = "Neutral"
        if sentiment_scores['compound'] > 0.05:
            sentiment_label = "Positive"
        elif sentiment_scores['compound'] < -0.05:
            sentiment_label = "Negative"

        user_input_news = translate_to_igbo(user_query)
        tavily_answer = translate_to_igbo(tavily_answer)
        news_result = Factcheck.objects.create(
        user_input_news=user_query,
        fresult=tavily_answer,
        sentiment_label=sentiment_label,
        genuine_urls=genuine_urls,
        non_authentic_urls=non_authentic_urls,
        num_genuine_sources= num_genuine_sources,
        genuine_urls_and_dates=genuine_urls_and_dates,
        non_authentic_sources = non_legit_length

        )
        # Redirect to the result page with primary key
        return redirect('result_detail_igbo', slug=news_result.slug)

    return render(request, 'igbo_factcheck.html')

def result_detail_igbo(request, slug):
    # Retrieve Factcheck object using primary key
    factcheck = Factcheck.objects.get(slug=slug)

    return render(request, 'igbo_preview.html', {
        'factcheck': factcheck,
        'genuine_urls': factcheck.genuine_urls,
        'non_authentic_urls': factcheck.non_authentic_urls,
        'genuine_urls_and_dates': factcheck.genuine_urls_and_dates,
        'num_genuine_sources': factcheck.num_genuine_sources,
        'user_input_news': factcheck.user_input_news,
        'sentiment_label': factcheck.sentiment_label,
        'tavily_answer': factcheck.fresult,
        'n':factcheck.non_authentic_sources
        
    })


def swahili_result(request):
    if request.method == 'POST':
        start_time = time.time()
        # Check if the request contains speech input
        speech_input_result = request.POST.get('speechInputResult', '')
        hidden_input_value = request.POST.get('hiddenInput', '')
        if speech_input_result:
            # If speech input is provided, use it as user_query
            user_query = speech_input_result
            is_speech_input = True
        else:
            # If no speech input, use the text input as user_query
            user_query = request.POST.get('query', '')
            is_speech_input = False
        
        tavily_answer = get_tavily_answer_with_retry(user_query)
        authenticity_result, genuine_sources, num_genuine_sources, genuine_urls, non_authentic_urls, non_authentic_sources,non_legit_length = check_news_authenticity(user_query)

        genuine_urls_and_dates = {}
        for url in genuine_urls:
            try:
                html = requests.get(url).content.decode('utf-8')
                publication_date = find_date(html)
                genuine_urls_and_dates[url] = publication_date
            except Exception as e:
                print(f"Error fetching date for {url}: {e}")
                genuine_urls_and_dates[url] = None


        user_input_news = user_query

        sentiment_scores = analyzer.polarity_scores(user_query)
        sentiment_label = "Neutral"
        if sentiment_scores['compound'] > 0.05:
            sentiment_label = "Positive"
        elif sentiment_scores['compound'] < -0.05:
            sentiment_label = "Negative"

        user_input_news = translate_to_swahili(user_query)
        tavily_answer = translate_to_swahili(tavily_answer)
        # Save factcheck result with primary key
        news_result = Factcheck.objects.create(
            user_input_news=user_query,
            fresult=tavily_answer,
            sentiment_label=sentiment_label,
            genuine_urls=genuine_urls,
            non_authentic_urls=non_authentic_urls,
            num_genuine_sources= num_genuine_sources,
            genuine_urls_and_dates=genuine_urls_and_dates,
            non_authentic_sources = non_legit_length

        )
        return redirect('result_detail_swahili', slug=news_result.slug)

    return render(request, 'swahili_factcheck.html')

def result_detail_swahili(request, slug):
    # Retrieve Factcheck object using primary key
    factcheck = Factcheck.objects.get(slug=slug)

    return render(request, 'swahili_preview.html', {
        'factcheck': factcheck,
        'genuine_urls': factcheck.genuine_urls,
        'non_authentic_urls': factcheck.non_authentic_urls,
        'genuine_urls_and_dates': factcheck.genuine_urls_and_dates,
        'num_genuine_sources': factcheck.num_genuine_sources,
        'user_input_news': factcheck.user_input_news,
        'sentiment_label': factcheck.sentiment_label,
        'tavily_answer': factcheck.fresult,
        'n':factcheck.non_authentic_sources
        
    })


def french_result(request):
    if request.method == 'POST':
        start_time = time.time()
        # Check if the request contains speech input
        speech_input_result = request.POST.get('speechInputResult', '')
        hidden_input_value = request.POST.get('hiddenInput', '')
        if speech_input_result:
            # If speech input is provided, use it as user_query
            user_query = speech_input_result
            is_speech_input = True
        else:
            # If no speech input, use the text input as user_query
            user_query = request.POST.get('query', '')
            is_speech_input = False
        
        tavily_answer = get_tavily_answer_with_retry(user_query)
        authenticity_result, genuine_sources, num_genuine_sources, genuine_urls, non_authentic_urls, non_authentic_sources,non_legit_length = check_news_authenticity(user_query)

        genuine_urls_and_dates = {}
        for url in genuine_urls:
            try:
                html = requests.get(url).content.decode('utf-8')
                publication_date = find_date(html)
                genuine_urls_and_dates[url] = publication_date
            except Exception as e:
                print(f"Error fetching date for {url}: {e}")
                genuine_urls_and_dates[url] = None


        user_input_news = user_query

        sentiment_scores = analyzer.polarity_scores(user_query)
        sentiment_label = "Neutre"
        if sentiment_scores['compound'] > 0.05:
            sentiment_label = "Positif"
        elif sentiment_scores['compound'] < -0.05:
            sentiment_label = "Négatif"

        user_input_news = translate_to_french(user_query)
        tavily_answer = translate_to_french(tavily_answer)
        # Save factcheck result with primary key
        news_result = Factcheck.objects.create(
            user_input_news=user_query,
            fresult=tavily_answer,
            sentiment_label=sentiment_label,
            genuine_urls=genuine_urls,
            non_authentic_urls=non_authentic_urls,
            num_genuine_sources= num_genuine_sources,
            genuine_urls_and_dates=genuine_urls_and_dates,
            non_authentic_sources = non_legit_length

        )
        return redirect('result_detail_french', slug=news_result.slug)

    return render(request, 'french_factcheck.html')

def result_detail_french(request, slug):
    # Retrieve Factcheck object using primary key
    factcheck = Factcheck.objects.get(slug=slug)

    return render(request, 'french_preview.html', {
        'factcheck': factcheck,
        'genuine_urls': factcheck.genuine_urls,
        'non_authentic_urls': factcheck.non_authentic_urls,
        'genuine_urls_and_dates': factcheck.genuine_urls_and_dates,
        'num_genuine_sources': factcheck.num_genuine_sources,
        'user_input_news': factcheck.user_input_news,
        'sentiment_label': factcheck.sentiment_label,
        'tavily_answer': factcheck.fresult,
        'n':factcheck.non_authentic_sources
        
    })


def arabic_result(request,):
    if request.method == 'POST':
        start_time = time.time()
        # Check if the request contains speech input
        speech_input_result = request.POST.get('speechInputResult', '')
        hidden_input_value = request.POST.get('hiddenInput', '')
        if speech_input_result:
            # If speech input is provided, use it as user_query
            user_query = speech_input_result
            is_speech_input = True
        else:
            # If no speech input, use the text input as user_query
            user_query = request.POST.get('query', '')
            is_speech_input = False
        
        tavily_answer = get_tavily_answer_with_retry(user_query)
        authenticity_result, genuine_sources, num_genuine_sources, genuine_urls, non_authentic_urls, non_authentic_sources,non_legit_length = check_news_authenticity(user_query)

        genuine_urls_and_dates = {}
        for url in genuine_urls:
            try:
                html = requests.get(url).content.decode('utf-8')
                publication_date = find_date(html)
                genuine_urls_and_dates[url] = publication_date
            except Exception as e:
                print(f"Error fetching date for {url}: {e}")
                genuine_urls_and_dates[url] = None


        user_input_news = user_query

        sentiment_scores = analyzer.polarity_scores(user_query)
        sentiment_label = "حيادي"
        if sentiment_scores['compound'] > 0.05:
            sentiment_label = "إيجابي"
        elif sentiment_scores['compound'] < -0.05:
            sentiment_label = "سلبي"

        user_input_news = translate_to_arabic(user_query)
        tavily_answer = translate_to_arabic(tavily_answer)
        # Save factcheck result with primary key
        news_result = Factcheck.objects.create(
            user_input_news=user_query,
            fresult=tavily_answer,
            sentiment_label=sentiment_label,
            genuine_urls=genuine_urls,
            non_authentic_urls=non_authentic_urls,
            num_genuine_sources= num_genuine_sources,
            genuine_urls_and_dates=genuine_urls_and_dates,
            non_authentic_sources = non_legit_length

        )

        
        # Redirect to the result page with primary key
        return redirect('result_detail_arabic', slug=news_result.slug)

    return render(request, 'arabic_factcheck.html')

def result_detail_arabic(request, slug):
    # Retrieve Factcheck object using primary key
    factcheck = Factcheck.objects.get(slug=slug)

    return render(request, 'arabic_preview.html', {
        'factcheck': factcheck,
        'genuine_urls': factcheck.genuine_urls,
        'non_authentic_urls': factcheck.non_authentic_urls,
        'genuine_urls_and_dates': factcheck.genuine_urls_and_dates,
        'num_genuine_sources': factcheck.num_genuine_sources,
        'user_input_news': factcheck.user_input_news,
        'sentiment_label': factcheck.sentiment_label,
        'tavily_answer': factcheck.fresult,
        'n':factcheck.non_authentic_sources
        
    })





def reliable_sources(request, user_query):
    authenticity_result, genuine_sources, num_genuine_sources, genuine_urls, non_authentic_urls, non_authentic_sources = check_news_authenticity(user_query)

    # Extract keywords for each URL
    url_keywords_mapping = {}
    keyword_sentiments = {}  # Store sentiment scores for each keyword
    for url in genuine_urls:
        parsed_url = urlparse(url)
        path = parsed_url.path
        keywords = path.split('/')

        # Filter out empty keywords and extract the last part of the path
        non_empty_keywords = [keyword.replace("-", " ") for keyword in keywords if keyword]

        if non_empty_keywords:
            # Extract the last part of the path as the keyword
            keyword = non_empty_keywords[-1]
            url_keywords_mapping[url] = keyword

            # Perform sentiment analysis on the keyword
            sentiment_scores = analyzer.polarity_scores(keyword)
            keyword_sentiments[keyword] = sentiment_scores
        else:
            # If no non-empty keywords found, set an empty string
            url_keywords_mapping[url] = ""

    # Perform sentiment analysis on the user's input
    sentiment_scores = analyzer.polarity_scores(user_query)

    # Determine sentiment label
    sentiment_label = "Neutral"
    if sentiment_scores['compound'] > 0.05:
        sentiment_label = "Positive"
    elif sentiment_scores['compound'] < -0.05:
        sentiment_label = "Negative"

    return render(request, 'reliable.html', {'genuine_sources': genuine_sources,
                                              'result': authenticity_result,
                                              'genuine_urls': genuine_urls,
                                              'num_genuine_sources': num_genuine_sources,
                                              'non_authentic_urls': non_authentic_urls,
                                              'n': non_authentic_sources,
                                              'user_input_news': user_query,
                                              'url_keywords_mapping': url_keywords_mapping,
                                              'keyword_sentiments': keyword_sentiments,
                                              'sentiment_label': sentiment_label})






def fact1(request):
    events=Factcheck.objects.all()
    print(events)
    context={'events':events}
    return render(request,'fact.html',context)

# Tavily and LLM setup
def get_tavily_answer_with_retry(query):
    # Initialize Tavily search and agent
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
    search = TavilySearchAPIWrapper()
    tavily_tool = TavilySearchResults(api_wrapper=search)

    # Create the agent
    agent_chain = initialize_agent(
        [tavily_tool],
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # Run the agent with the user query
    output = agent_chain.run(query)
    return output

class FactcheckAPI(APIView):
    def post(self, request):
        user_query = request.data.get('user_input_news', '')

        # Use Tavily to get the answer
        tavily_answer = get_tavily_answer_with_retry(user_query)

        # Check news authenticity
        authenticity_result, genuine_sources, num_genuine_sources, genuine_urls, non_authentic_urls, non_authentic_sources, non_legit_length = check_news_authenticity(user_query)

        # Extract dates for genuine URLs
        genuine_urls_and_dates = {}
        for url in genuine_urls:
            try:
                html = requests.get(url).content.decode('utf-8')
                publication_date = find_date(html)
                genuine_urls_and_dates[url] = publication_date
            except Exception as e:
                print(f"Error fetching date for {url}: {e}")
                genuine_urls_and_dates[url] = None

        # Perform sentiment analysis
        sentiment_scores = analyzer.polarity_scores(user_query)
        sentiment_label = "Neutral"
        if sentiment_scores['compound'] > 0.05:
            sentiment_label = "Positive"
        elif sentiment_scores['compound'] < -0.05:
            sentiment_label = "Negative"

        # Save the factcheck result to the database
        factcheck = Factcheck.objects.create(
            user_input_news=user_query,
            fresult=tavily_answer,
            sentiment_label=sentiment_label,
            genuine_urls=genuine_urls,
            non_authentic_urls=non_authentic_urls,
            num_genuine_sources=num_genuine_sources,
            genuine_urls_and_dates=genuine_urls_and_dates,
            non_authentic_sources=non_legit_length
        )

        serializer = FactcheckSerializer(factcheck)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    



# os.environ["SERPER_API_KEY"] = "fee3f70571c3c4be9d5d3e193e0946fa169c6405"

# from django.shortcuts import render
# from django.http import JsonResponse
# from crewai import Agent, Task, Crew
# from crewai_tools import SerperDevTool

# # Define the intelligent assistant
# intelligent_agent = Agent(
#     role='Conversational and Fact-Checking Assistant',
#     goal="""Act as both a friendly conversational assistant and a fact-checking expert. 
#     - If the user input is a greeting or casual conversation, respond conversationally and naturally.
#     - If the user input involves a claim or request for fact-checking, analyze the claim, verify its authenticity, and provide a detailed analysis, including source links.""",
#     backstory="""You're a dual-purpose AI assistant with expertise in engaging conversations and fact-checking. 
#     Your primary goals are:
#     - Responding to greetings or casual inquiries in a friendly and informative manner.
#     - Analyzing and verifying claims to detect misinformation using credible sources, and providing clear references with source links.""",
#     verbose=True
# )

# # Initialize the semantic search tool for fact-checking
# search_tool = SerperDevTool()

# def process_user_input(request):
#     if request.method == 'POST':
#         user_input = request.POST.get('user_input')

#         # Define the task dynamically based on user input
#         task = Task(
#             description=f"Process the user input: '{user_input}'",
#             expected_output="""Provide a response based on the type of input:
#             - For greetings or casual conversation, generate a natural, friendly response.
#             - For claims or fact-checking requests, provide:
#               - Whether the claim is True, False, or Unverified.
#               - A summary of evidence supporting the conclusion.
#               - References to credible sources, including their URLs, used for verification.""",
#             agent=intelligent_agent,
#             tools=[search_tool]
#         )

#         # Create a crew with the intelligent agent and task
#         crew = Crew(
#             agents=[intelligent_agent],
#             tasks=[task],
#             verbose=True
#         )

#         # Execute the task and get the result
#         result = crew.kickoff()

#         # Safely handle CrewOutput object
#         try:
#             # Assuming result has an attribute 'output' or similar
#             output_data = str(result)  # Convert the CrewOutput to a string if it cannot be serialized directly
#             return JsonResponse({'result': output_data})
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)

#     # Render the input page
#     return render(request, 'input_page.html')


# class FactCheckAPIView(APIView):
#     def post(self, request):
#         user_input = request.data.get('user_input', '')

#         if not user_input:
#             return Response(
#                 {"error": "User input is required."},
#                 status=status.HTTP_400_BAD_REQUEST
#             )

#         # Define the task dynamically based on user input
#         task = Task(
#             description=f"Process the user input: '{user_input}'",
#             expected_output="""Provide a response based on the type of input:
#             - For greetings or casual conversation, generate a natural, friendly response.
#             - For claims or fact-checking requests, provide:
#               - Whether the claim is True, False, or Unverified.
#               - A summary of evidence supporting the conclusion.
#               - References to credible sources, including their URLs, used for verification.""",
#             agent=intelligent_agent,
#             tools=[search_tool]
#         )

#         # Create a crew with the intelligent agent and task
#         crew = Crew(
#             agents=[intelligent_agent],
#             tasks=[task],
#             verbose=True
#         )

#         # Execute the task and handle the result
#         try:
#             result = crew.kickoff()
#             output_data = str(result)  # Convert CrewOutput to string if not serializable
#             return Response({'result': output_data}, status=status.HTTP_200_OK)
#         except Exception as e:
#             return Response(
#                 {"error": f"An error occurred while processing the request: {str(e)}"},
#                 status=status.HTTP_500_INTERNAL_SERVER_ERROR
#             )



# from django.shortcuts import render
# from crewai import Agent, Task, Crew
# from crewai_tools import ScrapeWebsiteTool, SerperDevTool

# def factcheck_webpage(request):
#     if request.method == 'POST':
#         # Get the URL from the user input
#         webpage_url = request.POST.get('webpage_url', '').strip()
        
#         if not webpage_url:
#             return render(request, 'factcheck_webpage.html', {'error': 'Please provide a valid webpage URL.'})
        
#         # Initialize the website scraping tool
#         scrape_tool = ScrapeWebsiteTool(website_url=webpage_url)

#         # Initialize the fact-checking tool
#         fact_check_tool = SerperDevTool()

#         # Define the fact-checking agent
#         fact_checker_agent = Agent(
#             role="Fact-Checking Expert",
#             goal="Extract claims from a webpage, verify them for authenticity, and detect misinformation.",
#             backstory="""You are an expert in detecting misinformation and fact-checking claims.
#             Your task is to analyze text content, extract notable claims, and verify their authenticity.""",
#             verbose=True
#         )

#         # Define the claim extraction and fact-checking task
#         extraction_task = Task(
#             description="Extract notable claims from the webpage content and verify them for authenticity.",
#             expected_output="""A report that includes:
#             - Extracted claims (if any) from the content
#             - Verification results for each claim
#             - References to credible sources for verification
#             - If no claims are found, a message stating 'No claims to fact-check'.""",
#             agent=fact_checker_agent,
#             tools=[scrape_tool, fact_check_tool]
#         )

#         # Define the crew with the agent and task
#         crew = Crew(
#             agents=[fact_checker_agent],
#             tasks=[extraction_task],
#             verbose=True
#         )

#         try:
#             # Run the task
#             result = crew.kickoff()

#             # Process and display results
#             if result:
#                 return render(request, 'factcheck_webpage.html', {'result': result})
#             else:
#                 return render(request, 'factcheck_webpage.html', {'error': 'No claims to fact-check.'})
#         except Exception as e:
#             return render(request, 'factcheck_webpage.html', {'error': f'Error processing the webpage: {str(e)}'})

#     # Render the input page
#     return render(request, 'factcheck_webpage.html')


# from django.shortcuts import render
# from django.http import JsonResponse
# from crewai import Agent, Task, Crew
# from crewai_tools import SerperDevTool, ScrapeWebsiteTool

# # Define the intelligent assistant
# intelligent_agent = Agent(
#     role='Conversational and Fact-Checking Assistant',
#     goal="""Act as both a friendly conversational assistant and a fact-checking expert. 
#     - If the user input is a greeting or casual conversation, respond conversationally and naturally.
#     - If the user input involves a claim or request for fact-checking, analyze the claim, verify its authenticity, and provide a detailed analysis, including source links.""",
#     backstory="""You're a dual-purpose AI assistant with expertise in engaging conversations and fact-checking. 
#     Your primary goals are:
#     - Responding to greetings or casual inquiries in a friendly and informative manner.
#     - Analyzing and verifying claims to detect misinformation using credible sources, and providing clear references with source links.""",
#     verbose=True
# )

# # Initialize the semantic search tool for fact-checking
# search_tool = SerperDevTool()

# def combined_view(request):
#     context = {}

#     if request.method == 'POST':
#         # Handle user input form
#         if 'user_input' in request.POST:
#             user_input = request.POST.get('user_input')

#             # Define the task dynamically based on user input
#             task = Task(
#                 description=f"Process the user input: '{user_input}'",
#                 expected_output="""Provide a response based on the type of input:
#                 - For greetings or casual conversation, generate a natural, friendly response.
#                 - For claims or fact-checking requests, provide:
#                   - Whether the claim is True, False, or Unverified.
#                   - A summary of evidence supporting the conclusion.
#                   - References to credible sources, including their URLs, used for verification.""",
#                 agent=intelligent_agent,
#                 tools=[search_tool]
#             )

#             # Create a crew with the intelligent agent and task
#             crew = Crew(
#                 agents=[intelligent_agent],
#                 tasks=[task],
#                 verbose=True
#             )

#             try:
#                 # Execute the task
#                 result = crew.kickoff()
#                 context['user_input_result'] = str(result)  # Convert CrewOutput to string
#             except Exception as e:
#                 context['user_input_error'] = f"Error processing input: {str(e)}"

#         # Handle webpage URL form
#         if 'webpage_url' in request.POST:
#             webpage_url = request.POST.get('webpage_url', '').strip()

#             if not webpage_url:
#                 context['webpage_error'] = 'Please provide a valid webpage URL.'
#             else:
#                 # Initialize the website scraping tool
#                 scrape_tool = ScrapeWebsiteTool(website_url=webpage_url)

#                 # Define the fact-checking agent
#                 fact_checker_agent = Agent(
#                     role="Fact-Checking Expert",
#                     goal="Extract claims from a webpage, verify them for authenticity, and detect misinformation.",
#                     backstory="""You are an expert in detecting misinformation and fact-checking claims.
#                     Your task is to analyze text content, extract notable claims, and verify their authenticity.""",
#                     verbose=True
#                 )

#                 # Define the claim extraction and fact-checking task
#                 extraction_task = Task(
#                     description="Extract notable claims from the webpage content and verify them for authenticity.",
#                     expected_output="""A report that includes:
#                     - Extracted claims (if any) from the content
#                     - Verification results for each claim
#                     - References to credible sources for verification
#                     - If no claims are found, a message stating 'No claims to fact-check'.""",
#                     agent=fact_checker_agent,
#                     tools=[scrape_tool, search_tool]
#                 )

#                 # Create the crew with the agent and task
#                 crew = Crew(
#                     agents=[fact_checker_agent],
#                     tasks=[extraction_task],
#                     verbose=True
#                 )

#                 try:
#                     # Run the task
#                     result = crew.kickoff()
#                     context['webpage_result'] = str(result)  # Convert CrewOutput to string
#                 except Exception as e:
#                     context['webpage_error'] = f"Error processing webpage: {str(e)}"

#     return render(request, 'combined_view.html', context)

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import os

# Set up API keys
os.environ["TAVILY_API_KEY"] = ""
os.environ['OPENAI_API_KEY'] = ""

# Set up the TavilySearchAPIRetriever for retrieving fact-based information
retriever = TavilySearchAPIRetriever(k=5)  # Retrieve top 5 relevant documents

# Custom prompt tailored for direct fact-checking with greetings handling
prompt = ChatPromptTemplate.from_template(
    """
    You are a Fact-Checking AI assistant. Your primary role is to verify claims or statements and provide accurate, fact-based responses. However, you should respond warmly and engagingly to greetings.

    Response Guidelines:
    - If the user greets (e.g., says "hello," "hi," or similar), respond in a friendly and engaging manner.
    - For all other inputs, check if the input is a claim or question requiring fact-checking:
      - If yes, provide a concise and factual response based on the given context.
      - If no, politely inform the user that you are a fact-checking assistant and can only verify claims or statements requiring verification.

    Context: {context}
    
    Question: {question}
    
    Response Format:
    - For greetings: Provide a friendly and engaging response.
    - For fact-checking: Provide a direct, fact-based answer, avoiding unnecessary qualifiers.
    - For unrelated queries: Politely explain your purpose and limitations.
    """
)

# Define the language model with minimal configuration
llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0,  # Ensure deterministic responses
)

# Format retrieved documents into a readable string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the processing chain
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class FactCheckWithTavilyAPIView(APIView):
    def post(self, request):
        user_input = request.data.get('user_input', '')

        if not user_input:
            return Response(
                {"error": "User input is required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Process the user input through the chain
        try:
            response = chain.invoke(user_input)
            return Response({"response": response}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": f"An error occurred while processing the request: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
