import spacy
import json
import torch

from torch_geometric.data import Data

class Grapher():

    def __init__(self, dep_map_path="./src/process/dep_map.json", tag_map_path="./src/process/tag_map.json"):
        self.parser = spacy.load('./data/spacy_model/en_core_web_sm-2.2.5')

        with open(dep_map_path, 'r') as f:
            dep_map = json.load(f)
        assert type(dep_map) == dict
        self.dep_map = dep_map["dep_to_id"]

        with open(tag_map_path, 'r') as f:
            tag_map = json.load(f)
        assert type(tag_map) == dict
        self.tag_map = tag_map["tag_to_id"]

        self.distance_gate = 1

    def build_dep_graph_from_text(self, text):

        text = text.lower()
        doc = self.parser(text)

        word_list = text.split()
        index_map = self.get_index_map(word_list, doc)

        edge_idx = []
        edge_type = []

        word_tags = dict()

        for index, token in enumerate(doc):

            word_tag = token.tag_
            if word_tag in self.tag_map.keys():
                tag_idx = self.tag_map[word_tag]
            else:
                tag_idx = 0
            word_tags[index_map[token.i]] = tag_idx

            if index_map[token.i] != index_map[token.head.i]:
                dep = token.dep_
                out_index = index_map[token.head.i]
                in_index = index_map[token.i]

                if dep in ['nsubj', 'advmod', 'attr']:
                    out_index, in_index = in_index, out_index

                if dep in self.dep_map.keys():
                    dep_idx = self.dep_map[dep]
                else:
                    dep_idx = 0

                edge_idx.append([out_index, in_index])
                edge_type.append([dep_idx])

        word_tags = [word_tags[i] for i in range(len(word_list))]

        assert len(word_tags) == len(word_list)
        return edge_idx, edge_type, word_tags

    def build_distance_graph_from_text(self, text):

        edge_idx = []
        edge_type = []
        edge_distace = []

        dep_graph_edge_idx, dep_graph_edge_type, word_tags = self.build_dep_graph_from_text(text)

        word_list = text.split()

        for i, word in enumerate(word_list):
            for j, word in enumerate(word_list):
                if i!=j:
                    distance = i-j
                    dep = [0]
                    distance_gate = self.distance_gate

                    if [i, j] in dep_graph_edge_idx:
                        temp = dep_graph_edge_idx.index([i, j])
                        dep = dep_graph_edge_type[temp]
                        if abs(distance) >= distance_gate:
                            distance = distance_gate if distance > 0 else -distance_gate

                    if abs(distance) < distance_gate or dep != [0]:
                    # if abs(distance) < distance_gate:

                        edge_idx.append([i, j])
                        edge_type.append(dep)
                        edge_distace.append([distance+5])

        return edge_idx, edge_type, edge_distace, word_tags

    def get_index_map(self, word_list, doc):
        '''
        text: origin text
        doc: parsered text (token list)
        '''

        if len(word_list)!=len(doc):
            index_map = []
            word_list_index = 0
            tmp = False
            sub_word = ""
            for index, token in enumerate(doc):
                if token.text == word_list[word_list_index]:
                    index_map.append(word_list_index)
                    word_list_index += 1
                    tmp=False

                elif token.text in word_list[word_list_index]:
                    if not tmp:
                        sub_word = ""
                        sub_word += token.text
                        index_map.append(word_list_index)
                        tmp = True

                    else:
                        sub_word += token.text
                        if sub_word in word_list[word_list_index]:
                            index_map.append(word_list_index)
                            tmp = True

                        else:
                            assert token.text in word_list[word_list_index + 1]
                            word_list_index += 1
                            index_map.append(word_list_index)
                            word_list_index += 1
                            tmp = False

                elif word_list_index + 1 < len(word_list) and token.text == word_list[word_list_index + 1]:
                    word_list_index += 1
                    index_map.append(word_list_index)
                    word_list_index += 1
                    tmp=False
                elif word_list_index + 1 < len(word_list):
                    word_list_index += 1
                    index_map.append(word_list_index)
                    tmp=False

        else:
            index_map = range(len(doc))

        # print(len(doc), len(word_list), word_list)
        # print(index_map)

        return index_map


    def get_graph(self, text, graph_type="distance+dep"):

        if graph_type == "dep":
            edge_idx, edge_type, word_tags = self.build_dep_graph_from_text(text)

            edge_index = torch.tensor(edge_idx, dtype=torch.long).t().contiguous()
            edge_type = torch.tensor(edge_type, dtype=torch.long)

            edge_data = Data(edge_idx=edge_index, edge_type=edge_type)

        else:
            edge_idx, edge_type, edge_distace, word_tags = self.build_distance_graph_from_text(text)
            edge_index = torch.tensor(edge_idx, dtype=torch.long).t().contiguous()
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_distace = torch.tensor(edge_distace, dtype=torch.long)

            edge_data = Data(edge_idx=edge_index, edge_type=edge_type, edge_distance=edge_distace)

        word_tags = torch.tensor(word_tags, dtype=torch.long)

        return [edge_data, word_tags]

if __name__ == "__main__":
    text = "Wine list selection is good and wine-by-the-glass was generously filled to the top ."
    grapher = Grapher()
    edge_data = grapher.get_graph(text, graph_type="distance+dep")
    print(edge_data)



