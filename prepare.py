from utils.gpt import get_attribute, get_des_question, get_binary_question
import json
attribute_dict = {}

# A description of your task:
description = """

"""
main_object = ""
attributes = get_attribute(description, main_object)
print("Pick some attributes:\n", attributes)
print("You can continue debug in run.ipynb")