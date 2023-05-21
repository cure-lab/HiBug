from utils.gpt import get_attribute, get_des_attribute, 
from processData.Corpus import emptyAttribute
import json
attribute_dict = {}

# A description of your task:
description = """

"""
main_object = ""
attributes = get_attribute(description, main_object)
print(attributes)

# Select some of the attributes
picked_attribute = []
types = []

for name,data_type in zip(picked_attribute,types):
    attribute_dict[name] = emptyAttribute()
    attribute_dict[name]['type'] = data_type
    if data_type == 'description':
        attribute_dict[name]['question'] = get_des_question(main_object,name)
    else:
        attribute_dict[name]['question'] = get_binary_question(main_object,name)

# Although GPT is amazing, it is better to double-check the corpus base
for key, content in attribute_dict.items():
    print(key, content)

f = open("exampleData/corpus_base.json" ,'w')
f.write(json.dumps(save_dict,indent=4))