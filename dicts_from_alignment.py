import argparse
from collections import Counter, defaultdict

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", help="comma-delimited dataset list", required=True, type=str
    )
    args = parser.parse_args()
    datasets = [dataset for dataset in args.datasets.split(",")]
    for dataset in datasets:
        src_to_dst = defaultdict(lambda: Counter())
        src_lang, dst_lang = dataset.split("-")
        with open(f"{dataset}/truecased/train.{dataset}.{src_lang}") as src_file, open(
            f"{dataset}/truecased/train.{dataset}.{dst_lang}"
        ) as dst_file, open(f"{dataset}/align/train.aligned") as align_file:
            for line_triplet in tqdm(zip(src_file, dst_file, align_file)):
                line_src, line_dst, line_align = map(
                    lambda x: x.strip().split(), line_triplet
                )
                for pair in line_align:
                    src_ind, dst_ind = pair.split("-")
                    src_to_dst[line_src[int(src_ind)]].update(
                        {line_dst[int(dst_ind)]: 1}
                    )
        with open(f"{dataset}/align/dict", "w+") as out_file:
            for word, counter in tqdm(src_to_dst.items()):
                print(f"{word} {counter.most_common(1)[0][0]}", file=out_file)


if __name__ == "__main__":
    main()
