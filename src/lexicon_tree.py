import collections

class Node():
    def __init__(self):
        self.next_node = collections.defaultdict(Node)
        self.is_word = False

class lexicon_tree():
    def __init__(self) -> None:
        self.root = Node()

    def insert_word(self, words : list):
        for word in words:
            if len(word) < 2:
                continue
            point = self.root
            for ch in word:
                point = point.next_node[ch]
            point.is_word = True

    def get_lattice(self, sentence : str):
        lattice = []
        sen_len = len(sentence)

        start_pos = [i for i in range(sen_len)]
        end_pos = [i for i in range(sen_len)]

        for i in range(len(sentence)):
            pointer = self.root
            for j, ch in enumerate(sentence[i:]):
                pointer = pointer.next_node.get(ch)
                if pointer is None:
                    break
                if pointer.is_word:
                    lattice.append((i, i+j, sentence[i:i+j+1]))

        sentence = list(str(sentence))

        for start, end, lat in lattice:
            start_pos.append(start)
            end_pos.append(end)
            sentence.append(lat)

        return [start_pos, end_pos, sentence, sen_len]
