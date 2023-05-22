import openai
import json

'''
# Ask chatGPT to slice the sentence
'''
api_key = ''
openai.api_key = api_key
assert api_key != '', 'Need gpt api'
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
        in a form: "description attributes": name1, name2, ... name10; "binary attributes": name1, name2, ... name10. The attributes should related to visual feature only.
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
        Give me a question asking if the %s in the photo %s or not. Return the question only. No other words. No quotation mark.
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
         Give me a question asking the %s of the %s in the photo. Return the question only. No other words. No quotation mark.
    """%(attribute,main_object)

    messages.append(
            {"role": "user", "content": question},
    )

    chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
    )
    reply = chat.choices[0].message.content
    return reply

def get_des_prompt(main_object, attribute):
    question = """
        Give me a sentence combining a visual feature and a main object. For example, when the feature is hair color, and the main object is people. 
        The sentence could be: A #LABEL with #1 hair. Here #LABEL is related to the main object. #1 represent a specific type of the visual feature. Your sentence also includes the #LABEL and #1. 
        Do not include other things in the sentence. It should be as simple as possible. Now the visual feature is %s, and the object is %s. Give me a sentence. No other words. No quotation mark.
    """%(attribute,main_object)

    messages.append(
            {"role": "user", "content": question},
    )

    chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
    )
    reply = chat.choices[0].message.content
    return reply