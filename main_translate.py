import argparse
import os
import os.path
import glob
import json
import xml.etree.ElementTree as ElementTree

import deep_translator
import translators as ts
from deep_translator import GoogleTranslator, DeeplTranslator
# requests = "~=2.27"
# python-lokalise-api = "~=1.6"
# python-dotenv = "~=0.20"
# googletrans = "==4.0.0rc1"
# translators = "~= 5.4"
# deep-translator = "~=1.9"

import main_clean as clean

def make_symbols():
    symbols = ["[]", "{}", "<>", "%%", "^^", "``", "~~", "○○" , "††", ]
    return symbols


def extract_marked_word(sentence: str, symbol: str):
    # print("extracting word from: " + sentence + "with symbol: " + symbol)
    substrings = sentence.split(symbol[0])
    substring1 = substrings[1]
    substrings2 = substring1.split(symbol[1])
    extraction = substrings2[0]
    return extraction


def mark_data(year:int, phase: str, language: str):
    print("Running marking")
    filename = f"ABSA{year % 2000}_Restaurants_{phase}_{language}.xml"
    filename_marked = f"ABSA{year % 2000}_Restaurants_{phase}_{language}Marked.xml"
    input_path = f"/content/drive/MyDrive/Thesis_Data/data/raw/{filename}"
    output_path = f"/content/drive/MyDrive/Thesis_Data/data/marked/{filename_marked}"
    tree = ElementTree.parse(input_path)
    symbols = make_symbols()

    for sentence in tree.findall(".//sentence"):
        text_element = sentence.find(".//text")
        sentence_text = text_element.text
        opinions = sentence.findall(".//Opinion")
        opinions.sort(key=lambda x: int(x.attrib['from']))

        last_start = ''
        k = 0
        for sorted_opinion in opinions:
            # If the sentence contains more aspects than the number of unique symbols, skip the aspect
            if k >= len(symbols):
                sorted_opinion.attrib['target'] = "NULL"
                continue

            symbol = symbols[k]
            start = int(sorted_opinion.attrib['from'])
            end = int(sorted_opinion.attrib['to'])

            if sorted_opinion.attrib['target'] == "NULL":
                continue

            print("target: " + sorted_opinion.attrib['target'] + " start: " + sorted_opinion.attrib['from'] + " last: " + last_start)
            if not sorted_opinion.attrib['from'] == last_start:
                start = start + 2*k
                end = end + 2*k

                sentence_text = sentence_text[:start] + symbol[0] + sentence_text[start:end] + symbol[1] + sentence_text[end:]

                sentence.find(".//text").text = sentence_text
                last_start = start.__str__()
                k += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tree.write(output_path)
    print("Marking done")


def translate_data(year:int, phase: str, source_language: str, target_language: str):
    print("Running translation")
    filename = f"ABSA{year % 2000}_Restaurants_{phase}_{source_language}Marked.xml"
    filename_translated = f"ABSA{year % 2000}_Restaurants_{phase}_{source_language}to{target_language}Translated.xml"
    input_path = f"/content/drive/MyDrive/Thesis_Data/data/marked/{filename}"
    output_path = f"/content/drive/MyDrive/Thesis_Data/data/translated/{filename_translated}"
    tree = ElementTree.parse(input_path)

    target = ''
    if target_language == "English":
        target = 'en'
    elif target_language == "Dutch":
        target = 'nl'
    elif target_language == "French":
        target = 'fr'
    elif target_language == "Spanish":
        target = 'es'

    translator = GoogleTranslator(source='auto', target=target)

    symbols = make_symbols()

    for sentence in tree.findall(".//sentence"):
            sentence_text = sentence.find(".//text").text
            # In case if the review only contains numeric values
            if not sentence_text:
                continue
            if sentence_text.isnumeric():
                continue

            translation = translator.translate(sentence_text)
            sentence.find(".//text").text = translation
            sentence_text = translation
            previous_positions = []
            previous_opinions = []
            double_opinions = []

            opinions = sentence.findall(".//Opinion")
            for opinion in opinions:
                position = [opinion.attrib['from'], opinion.attrib['to']]
                if previous_positions.__contains__(position):
                    opinion.attrib['from'] = "same as " + previous_positions.index(position).__str__()
                    double_opinions.append(opinion)
                else:
                    previous_opinions.append(opinion)
                    previous_positions.append(position)

            i = 0
            # Update aspect text with translation and update position
            for opinion in sentence.findall(".//Opinion"):
                if opinion.attrib['target'] == "NULL":
                    continue
                # Check if out of symbols
                if i >= len(symbols):
                    continue

                symbol = symbols[i]
                if double_opinions.__contains__(opinion):
                    continue

                # If brackets still in translation, extract aspect and update position
                elif sentence_text.__contains__(symbol[0]) and sentence_text.__contains__(symbol[1]):
                    aspect = extract_marked_word(sentence_text, symbol)
                    opinion.attrib['target'] = aspect
                    start = sentence_text.find(symbol[0]) + 1
                    end = sentence_text.rfind(symbol[1])
                    opinion.attrib['from'] = start.__str__()
                    opinion.attrib['to'] = end.__str__()

                    previous_positions.append([start.__str__(), end.__str__()])
                    previous_opinions.append(opinion)
                    i += 1

                else:
                    # sentence.findall(".//Opinion").remove(opinion)
                    opinion.attrib['target'] = "NULL"

                    i += 1

            for double in double_opinions:
                brother_index = int(double.attrib['from'][8])
                brother = previous_opinions[brother_index]
                double.attrib['target'] = brother.attrib['target']
                double.attrib['from'] = brother.attrib['from']
                double.attrib['to'] = brother.attrib['to']

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tree.write(output_path)


def aspect_code_switching(year: int, phase: str, source: str, target: str):
    print("Running ACS")
    filename_source = f"ABSA{year % 2000}_Restaurants_{phase}_{source}Marked.xml"
    filename_target = f"ABSA{year % 2000}_Restaurants_{phase}_{source}to{target}Translated.xml"
    filename_st = f"ABSA{year % 2000}_Restaurants_{phase}_{source}to{target}ACS.xml"
    filename_ts = f"ABSA{year % 2000}_Restaurants_{phase}_{target}to{source}ACS.xml"

    source_path = f"/content/drive/MyDrive/Thesis_Data/data/marked/{filename_source}"
    target_path = f"/content/drive/MyDrive/Thesis_Data/data/translated/{filename_target}"
    st_path = f"/content/drive/MyDrive/Thesis_Data/data/acs/{filename_st}"
    ts_path = f"/content/drive/MyDrive/Thesis_Data/data/acs/{filename_ts}"

    tree_source = ElementTree.parse(source_path)
    tree_target = ElementTree.parse(target_path)

    symbols = make_symbols()
    all_source_sentences = tree_source.findall(".//sentence")

    for sentence_source in all_source_sentences:
        source_text = sentence_source.find(".//text").text
        sentence_target = tree_target.findall(".//sentence")[all_source_sentences.index(sentence_source)]
        target_text = sentence_target.find(".//text").text

        # Switch each source aspect with each corresponding target aspect
        for symbol in symbols:

            # Only operates when sentence contains marking for aspect in source and target text
            if source_text.__contains__(symbol[0]) and target_text.__contains__(symbol[0]):
                source_aspect = extract_marked_word(source_text, symbol)
                target_aspect = extract_marked_word(target_text, symbol)

                # Replacing in source data
                # Replace source text with marked target aspect
                sentence_source.find(".//text").text = sentence_source.find(".//text").text.replace(source_aspect, target_aspect)

                # Replace source labels with target labels
                for opinion in sentence_source.findall(".//Opinion"):
                    # Make sure current aspect is equal to the one in the sentence
                    if opinion.attrib['target'] == "NULL":
                        continue

                    if opinion.attrib['target'] == source_aspect:
                        opinion.attrib['target'] = target_aspect

                # Replacing in target data
                # Replace target text with marked source aspect
                sentence_target.find(".//text").text = sentence_target.find(".//text").text.replace(target_aspect, source_aspect)

                # Replace target labels with source labels
                for opinion in sentence_target.findall(".//Opinion"):
                    if opinion.attrib['target'] == "NULL":
                        continue

                    if opinion.attrib['target'] == target_aspect:
                        opinion.attrib['target'] = source_aspect

    # Update positions in source to target data
    for sentence_source in tree_source.findall(".//sentence"):
        source_text = sentence_source.find(".//text").text
        previous_positions = []
        double_opinions = []
        new_positions = []
        for opinion in sentence_source.findall(".//Opinion"):
            if opinion.attrib["target"] == "NULL":
                continue

            # Checks whether position of an opinion has been seen before
            # If so, it saves it as a double opinion
            # If not it is added as a new opinion to the list of positions seen before
            position = [opinion.attrib['from'], opinion.attrib['to']]
            if previous_positions.__contains__(position):
                opinion.attrib['from'] = previous_positions.index(position).__str__()
                double_opinions.append(opinion)
            else:
                previous_positions.append(position)
        k = 0
        for opinion in sentence_source.findall(".//Opinion"):
            if opinion.attrib['target'] == "NULL":
                continue

            symbol = symbols[k]
            if not double_opinions.__contains__(opinion):
                new_start = source_text.find(symbol[0]) + 1
                new_end = source_text.rfind(symbol[1])
                opinion.attrib['from'] = new_start.__str__()
                opinion.attrib['to'] = new_end.__str__()
                new_positions.append([new_start.__str__(), new_end.__str__()])
                k += 1

        for double in double_opinions:
            brother_index = int(double.attrib['from'])
            brother_position = new_positions[brother_index]
            double.attrib['from'] = brother_position[0]
            double.attrib['to'] = brother_position[1]

    # Update positions in target to source data
    for sentence_target in tree_target.findall(".//sentence"):
        target_text = sentence_target.find(".//text").text
        previous_positions = []
        double_opinions = []
        new_positions = []
        for opinion in sentence_target.findall(".//Opinion"):
            if opinion.attrib['target'] == "NULL":
                continue

            position = [opinion.attrib['from'], opinion.attrib['to']]
            if previous_positions.__contains__(position):
                opinion.attrib['from'] = previous_positions.index(position).__str__()
                double_opinions.append(opinion)
            else:
                previous_positions.append(position)
        k = 0
        for opinion in sentence_target.findall(".//Opinion"):
            if opinion.attrib['target'] == "NULL":
                continue

            symbol = symbols[k]
            if target_text.__contains__(symbol[0]) and not double_opinions.__contains__(opinion):
                print("there is symbol: " + target_text)
                new_start = target_text.find(symbol[0]) + 1
                new_end = target_text.rfind(symbol[1])
                opinion.attrib['from'] = new_start.__str__()
                opinion.attrib['to'] = new_end.__str__()
                new_positions.append([new_start.__str__(), new_end.__str__()])
                k += 1
            elif not target_text.__contains__(symbol[0]):
                print("no symbol: " + target_text)
                k += 1

        for double in double_opinions:
            brother_index = int(double.attrib['from'])
            brother_position = new_positions[brother_index]
            double.attrib['from'] = brother_position[0]
            double.attrib['to'] = brother_position[1]

    os.makedirs(os.path.dirname(st_path), exist_ok=True)
    tree_source.write(st_path)
    tree_target.write(ts_path)


def remove_symbols(filename):
    tree = ElementTree.parse(filename)
    symbols = make_symbols()
    for sentence in tree.findall(".//sentence"):
        text_element = sentence.find(".//text")
        sentence_text = text_element.text
        if not sentence_text:
            continue

        already_changed = []
        opinions = sentence.findall(".//Opinion")
        # Find corresponding opinion

        opinions.sort(key=lambda x: int(x.attrib['from']))

        last_from = ''
        last_to = ''
        k = 0
        for opinion in opinions:
            if opinion.attrib['from'] != last_from and opinion.attrib['to'] != last_to:
                # Determine new aspect boundaries
                last_from = opinion.attrib['from']
                last_to = opinion.attrib['to']

                print("marked: " + opinion.attrib['from'])
                from_int = int(opinion.attrib['from']) - 1 - 2 * k
                to_int = int(opinion.attrib['to']) - 1 - 2 * k
                print("adjusted: " + from_int.__str__())
                opinion.attrib['from'] = from_int.__str__()
                opinion.attrib['to'] = to_int.__str__()


                k += 1
            else:
                # Same boundaries as previous
                opinion.attrib['from'] = from_int.__str__()
                opinion.attrib['to'] = to_int.__str__()

        # Remove symbols
        for symbol in symbols:
            sentence_text = sentence_text.replace(symbol[0], "", 1)
            sentence_text = sentence_text.replace(symbol[1], "", 1)

        sentence.find(".//text").text = sentence_text

    tree.write(filename)


def MLCR_Rot_hop_plus_plus(year, phase):

    english_path = f"/content/drive/MyDrive/Thesis_Data/data/processed/ABSA{year % 2000}_Restaurants_{phase}_English.xml"
    dutch_path = f"/content/drive/MyDrive/Thesis_Data/data/processed/ABSA{year % 2000}_Restaurants_{phase}_Dutch.xml"
    french_path = f"/content/drive/MyDrive/Thesis_Data/data/processed/ABSA{year % 2000}_Restaurants_{phase}_French.xml"
    spanish_path = f"/content/drive/MyDrive/Thesis_Data/data/processed/ABSA{year % 2000}_Restaurants_{phase}_Spanish.xml"

    multilingual_path = f"/content/drive/MyDrive/Thesis_Data/data/processed/ABSA{year % 2000}_Restaurants_{phase}_Multilingual.xml"

    xml_files = [english_path, dutch_path, french_path, spanish_path]

    root = ElementTree.Element("Reviews")
    forest = ElementTree.ElementTree(root)

    for file in xml_files:
        tree = ElementTree.parse(file)
        reviews = tree.findall(".//Review")
        root.extend(reviews)

    forest.write(multilingual_path)


def SLCR_Rot_hop_plus_plus(year, phase):
    english_path = f"/content/drive/MyDrive/Thesis_Data/data/processed/ABSA{year % 2000}_Restaurants_{phase}_English.xml"
    #This will need changing if adapted -> otherwise remove
    dutch_path = f"/content/drive/MyDrive/Thesis_Data/data/processed/ABSA{year % 2000}_Restaurants_{phase}_DutchTranslated.xml"
    french_path = f"/content/drive/MyDrive/Thesis_Data/data/processed/ABSA{year % 2000}_Restaurants_{phase}_FrenchTranslated.xml"
    spanish_path = f"/content/drive/MyDrive/Thesis_Data/data/processed/ABSA{year % 2000}_Restaurants_{phase}_SpanishTranslated.xml"

    multilingual_path = f"/content/drive/MyDrive/Thesis_Data/data/processed/ABSA{year % 2000}_Restaurants_{phase}_Translated.xml"

    xml_files = [english_path, dutch_path, french_path, spanish_path]

    root = ElementTree.Element("Reviews")
    forest = ElementTree.ElementTree(root)

    for file in xml_files:
        tree = ElementTree.parse(file)
        reviews = tree.findall(".//Review")
        root.extend(reviews)

    forest.write(multilingual_path)


def join_datasets_ACS(year, phase, source, target):
    filename_source = f"ABSA{year % 2000}_Restaurants_{phase}_{source}.xml"
    filename_target = f"ABSA{year % 2000}_Restaurants_{phase}_{source}to{target}Translated.xml"
    filename_st = f"ABSA{year % 2000}_Restaurants_{phase}_{source}to{target}ACS.xml"
    filename_ts = f"ABSA{year % 2000}_Restaurants_{phase}_{target}to{source}ACS.xml"

    filename_acs = f"ABSA{year % 2000}_Restaurants_{phase}_XACSfor{target}.xml"

    source_path = f"/content/drive/MyDrive/Thesis_Data/data/processed/{filename_source}"
    target_path = f"/content/drive/MyDrive/Thesis_Data/data/processed/{filename_target}"
    st_path = f"/content/drive/MyDrive/Thesis_Data/data/processed/{filename_st}"
    ts_path = f"/content/drive/MyDrive/Thesis_Data/data/processed/{filename_ts}"

    acs_path = f"/content/drive/MyDrive/Thesis_Data/data/processed/{filename_acs}"
    xml_files = [source_path, target_path, st_path, ts_path]

    root = ElementTree.Element("Reviews")
    forest = ElementTree.ElementTree(root)

    for file in xml_files:
        tree = ElementTree.parse(file)
        reviews = tree.findall(".//Review")
        root.extend(reviews)

    forest.write(acs_path)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--year", default=2016, type=int, help="The year of the dataset (2015 or 2016)")
    parser.add_argument("--phase", default="Train", help="The phase of the dataset (Train or Test)")
    parser.add_argument("--source", default="English", type=str, help="The language of the dataset")
    parser.add_argument("--target", default="Dutch", type=str, help="The target language")
    parser.add_argument("--model-type", default="MLCR-Rot-hop++", type = str, help = "Which type of model is data created for?")

    """
    Model types to select:
    1. MLCR-Rot-hop++
    2. mLCR-Rot-hop-XXen++
    3. mLCR-Rot-hop-ACSxx
    4. SLCR-Rot-hop++
    """

    args = parser.parse_args()

    year: int = args.year
    phase: str = args.phase
    source: str = args.source
    target: str = args.target
    model_type: str = args.model_type

    # Implementation based on instructions by Steiner: https://github.com/Anonymous71717/mLCR-Rot-hop-plus-plus.git
    if model_type == "MLCR-Rot-hop++":
        print ("Running MLCR-Rot-hop++...")
        MLCR_Rot_hop_plus_plus(year=year,phase=phase)
    #ACSxx uses the same methodology as XXen++ with extra steps
    elif model_type == "mLCR-Rot-hop-XXen++" or model_type == "mLCR-Rot-hop-ACSxx":
        print("Running " + model_type + "...")

        mark_data(year=year, phase=phase, language=source)
        translate_data(year=year, phase=phase, source_language=source, target_language=target)

        if model_type == "mLCR-Rot-hop-ACSxx":
            aspect_code_switching(year=year,phase=phase,source=source,target=target)

            remove_symbols(f"/content/drive/MyDrive/Thesis_Data/data/marked/ABSA{year % 2000}_Restaurants_{phase}_{source}Marked.xml")
            remove_symbols(f"/content/drive/MyDrive/Thesis_Data/data/acs/ABSA{year % 2000}_Restaurants_{phase}_{source}to{target}ACS.xml")
            remove_symbols(f"/content/drive/MyDrive/Thesis_Data/data/acs/ABSA{year % 2000}_Restaurants_{phase}_{target}to{source}ACS.xml")

            clean.clean_data(year=year,phase=phase,language=source + "Marked", dirname="marked")
            clean.clean_data(year=year,phase=phase,language=source+"to"+target+"ACS", dirname="acs")
            clean.clean_data(year=year,phase=phase,language=target+"to"+source+"ACS", dirname="acs")

        remove_symbols(f"/content/drive/MyDrive/Thesis_Data/data/translated/ABSA{year % 2000}_Restaurants_{phase}_{source}to{target}Translated.xml")
        clean.clean_data(year=year,phase=phase,language=source+"to"+target+"Translated",dirname="translated")

        if model_type == "mLCR-Rot-hop-ACSxx":
            join_datasets_ACS(year=year,phase=phase,source=source,target=target)
    else:
        print ("Invalid model type inputted")

if __name__ == "__main__":
    main()
