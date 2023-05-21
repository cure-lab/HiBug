import openai
import json

'''
# Ask chatGPT to slice the sentence
'''

openai.api_key = ''
messages = [ {"role": "system", "content":
        "You are a intelligent assistant."} ]

def get_attribute(description, main_object):
    question = """
        I define two kinds of dimension to describe image in deep learning visual task: "description attributes" and "binary attributes". 
        For example, in a human recognition task, where the main object in the image is human, 
        "description attributes" could be : 1.hair color 2.nose size; 
        "binary attributes" could be : 1.wearing necklace 2.with beard.
        The difference is that the description in "description attributes" is a pharse and the description in "binary attributes" is simply yes or no.
        Now, I have another task. Here is the description: %s. And here is the main object: %s.
        I need you to give me ten "description attributes" and ten "binary attributes" about this task. You only need to give me the name of the attribute, 
        in a form: "description attributes": name1, name2, ... name10; "binary attributes": name1, name2, ... name10.
    """%(description,main_object)

    messages.append(
            {"role": "user", "content": question},
    )

    chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
    )
    reply = chat.choices[0].message.content
    return reply

def get_des_question(main_object, attribute):
    question = """
         Give me a question asking the %s of the %s in the photo.
    """%(attribute,main_object)

    messages.append(
            {"role": "user", "content": question},
    )

    chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
    )
    reply = chat.choices[0].message.content
    return reply

def get_binary_question(main_object, attribute):
    question = """
        Give me a question asking if the %s in the photo %s or not.
    """%(main_object,attribute)

    messages.append(
            {"role": "user", "content": question},
    )

    chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
    )
    reply = chat.choices[0].message.content
    return reply

def get_des_question(main_object, attribute):
    question = """
         Give me a question asking the %s of the %s in the photo.
    """%(attribute,main_object)

    messages.append(
            {"role": "user", "content": question},
    )

    chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
    )
    reply = chat.choices[0].message.content
    return reply