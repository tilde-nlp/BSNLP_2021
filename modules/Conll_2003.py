import os

class Conll2003:
    def __init__(self):
        self.lines = [] # Intended to consist for tuples (word, label)
        self.labels = set() # Intended to contain all labels (with B- I- tags)
    
    def load_from_file(self, filepath):
        with open(filepath, 'r', encoding="utf-8") as inf:
            for line in inf:
                try:
                    text, label = line.strip().split(' ')
                except Exception as e:
                    text = ""
                    label = ""
                self.lines.append((text, label))


    def write_output(self, filepath):
        with open(filepath, 'w', encoding="utf-8") as outf:
            for line in self.lines:
                line2write = "{} {}".format(line[0],line[1])
                line2write = line2write.strip()
                outf.write("{}\n".format(line2write))

    def update_labels_file(self, filepath):
        # Intended to be used when writing new file to update labels.txt if 
        # there are new labels in current conll2003 file. 
        def load_labels(label_path):
            with open(label_path, 'r', encoding='utf-8') as labelfile:
                for line in labelfile.readlines():
                    line = line.strip()
                    self.labels.add(line)

        def extract_unseen_labels_from_current_file():
            new_labels = set()
            for line in self.lines:
                label = line[1]
                if label and label not in self.labels:
                    new_labels.add(label)
            return new_labels

        def save_labels(label_path):
            with open(label_path, 'w', encoding='utf-8') as labelfile:
                for label in self.labels:
                    if not label.strip():
                        continue
                    labelfile.write("{}\n".format(label))

        data_dir = os.path.dirname(filepath)
        label_path = os.path.join(data_dir, "labels.txt")
        try:
            load_labels(label_path)
        except:
            self.labels = set()
        new_labels = extract_unseen_labels_from_current_file()
        if new_labels:
            self.labels = self.labels.union(new_labels)
            save_labels(label_path)

