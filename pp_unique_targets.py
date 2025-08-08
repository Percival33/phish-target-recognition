def load_file_to_set(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def main():
    A = load_file_to_set("/home/phish-target-recognition/pp_targets.txt")
    B = load_file_to_set("/home/phish-target-recognition/pp_targets_occuring_in_vp.txt")
    diff = A.difference(B)
    for line in sorted(diff, key=lambda s: s.lower()):
        print(line)

if __name__ == "__main__":
    main()
