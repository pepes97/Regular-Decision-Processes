import json

class MM():

    def __init__(self, json_file):
        with open(json_file, "r") as jf:
            self.json = json.load(jf)
        self.current_node = '0'
        self.edges = {}  # {source : {name: target}}

    def build_mealy(self):
        for edge in self.json['edges']:
            source = edge["source"]
            target = edge["target"]
            name = edge["name"]
            if source not in self.edges:
                self.edges[source] = {name: target}
            else:
                self.edges[source][name] = target
        print(self.edges)
