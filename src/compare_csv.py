import argparse
import csv


def compare_csv(file1, file2):
    with open(file1, "r") as f1, open(file2, "r") as f2:
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)

        # calculate accuracy do not count header
        total = 0
        correct = 0
        for row1, row2 in zip(reader1, reader2):
            if row1[0] == "ID":
                continue
            total += 1
            if row1[0] != row2[0]:
                print(f"WARNING ID mismatch: {row1[0]} != {row2[0]}")
            if row1[1] == row2[1]:
                correct += 1

    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=str,
        help="Path to input file",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path to output directory",
    )

    args = parser.parse_args()

    print(f"Accuarcy: {compare_csv(args.input, args.output)}")
