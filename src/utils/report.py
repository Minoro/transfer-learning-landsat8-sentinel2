import os
import json
from types import ModuleType

class Reporter:

    def __init__(self):
        self.data = {}

    def add(self, values):
        if isinstance(values, ModuleType):
            attrs = dir(values)
            values = { attr : getattr(values, attr) for attr in attrs if not attr.startswith('__') }

        elif not type(values) == type(dict):
            values = vars(values)

        self.data = dict(self.data, **values)
    
    def push(self, key, value):
        self.data[key] = value

    def get(self):
        return self.data
    
    def reset(self):
        self.data = {}
    
    def to_json(self, output_file : str):
        if not output_file.endswith('.json'):
            output_file += '.json'
        
        with open(output_file, 'w+') as f:
           json.dump(self.data, f, default=str)