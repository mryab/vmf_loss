import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./", type=str)

    args = parser.parse_args()

    lg_pairs = ["de-en/", "en-fr/", "fr-en/"]

    for pair in lg_pairs:
        mypath = args.data_path + pair
        os.makedirs(mypath + "parsed", exist_ok=True)

        onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(mypath + f)]
        for file in onlyfiles:
            with open(mypath + file) as rd:
                with open(mypath + "parsed/" + file, "w") as wr:
                    for line in rd:
                        if line.startswith("<seg"):
                            wr.write(
                                line[line.find('">') + 2 : -len("</seg>\n")] + "\n"
                            )
                        if line[0] != "<":
                            wr.write(line)


if __name__ == "__main__":
    main()
