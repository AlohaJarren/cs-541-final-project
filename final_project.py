'''
Naive Bayes Pet Classifier
CS 441/541 Final Group Project
Jarren Calizo & Vivi Chen
'''

def load_data(filename="pets.txt"):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            label, text = line.split(" ", 1)
            data.append((label, text))
    return data

def main():
    data = load_data("pets.txt")
    print(f"Loaded {len(data)} pet descriptions.")
    print("First 5 samples:")
    for sample in data[:5]:
        print(sample)
        
if __name__ == "__main__":
    main()