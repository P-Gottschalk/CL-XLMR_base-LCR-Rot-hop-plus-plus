import argparse
import os
import xml.etree.ElementTree as ElementTree


def clean_data(year: int, phase: str, language: str, dirname: str):
    """Clean a SemEval dataset by removing opinions with implicit targets. This function returns the cleaned dataset."""
    filename = f"ABSA{year % 2000}_Restaurants_{phase}_{language}.xml"

    input_path = f"/content/drive/MyDrive/Thesis_Data/data/{dirname}/{filename}"
    output_path = f"/content/drive/MyDrive/Thesis_Data/data/processed/{filename}"

    # if os.path.isfile(output_path):
    #     print(f"Found cleaned file at {output_path}")
    #     return ElementTree.parse(output_path)

    tree = ElementTree.parse(input_path)

    # Remove invalid targets, i.e., with incorrect position
    n_invalid_removed = 0
    # remove implicit targets
    n_null_removed = 0
    for opinions in tree.findall(".//Opinions"):
        for opinion in opinions.findall('./Opinion[@from="-1"]'):
            opinions.remove(opinion)
            n_invalid_removed += 1
        for opinion in opinions.findall('./Opinion[@to="-1"]'):
            opinions.remove(opinion)
            n_invalid_removed += 1
        for opinion in opinions.findall('./Opinion[@target="NULL"]'):
            opinions.remove(opinion)
            n_null_removed += 1
    n_total_removed = n_invalid_removed + n_null_removed

    # calculate descriptive statistics for remaining opinions
    n = 0
    n_positive = 0
    n_negative = 0
    n_neutral = 0
    for opinion in tree.findall(".//Opinion"):
        n += 1

        if opinion.attrib['polarity'] == "positive":
            n_positive += 1
        elif opinion.attrib['polarity'] == "negative":
            n_negative += 1
        elif opinion.attrib['polarity'] == "neutral":
            n_neutral += 1

    if n == 0:
        print(f"\n{filename} does not contain any opinions")
    else:
        print(f"\n{filename}")
        print(f" Removed  {n_invalid_removed} opinions with invalid position")
        print(f"  Removed {n_null_removed} opinions with target NULL")
        print(f" Total removed: {n_total_removed}")
        print(f"  Total number of opinions remaining: {n}")
        print(f"  Fraction positive: {100 * n_positive / n:.3f} %")
        print(f"  Fraction negative: {100 * n_negative / n:.3f} %")
        print(f"  Fraction neutral: {100 * n_neutral / n:.3f} %")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tree.write(output_path)
    print(f"Stored cleaned dataset in {output_path}")

    return tree


def main():
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--year", default=2016, type=int, help="The year of the dataset (2015 or 2016)")
    parser.add_argument("--phase", default="Train", help="The phase of the dataset (Train or Test)")
    parser.add_argument("--language", default="English", type=str, help="The language of the dataset")
    parser.add_argument("--dirname", default="raw", type=str, help="The language of the dataset")
    args = parser.parse_args()

    year: int = args.year
    phase: str = args.phase
    language: str = args.language
    dirname: str = args.dirname

    clean_data(year, phase, language, dirname)


if __name__ == "__main__":
    main()
