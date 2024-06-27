import xml.etree.ElementTree as ElementTree

def make_symbols():
    symbols = ["[]", "{}", "<>", "%%", "^^", "``", "~~", "○○"]
    return symbols

def remove_symbols(filename):
    tree = ElementTree.parse(filename)
    symbols = make_symbols()
    for sentence in tree.findall(".//sentence"):
        text_element = sentence.find(".//text")
        sentence_text = text_element.text

        opinions = sentence.findall(".//Opinion")
        for symbol in symbols:
            # Find corresponding opinion
            opinion = opinions[symbols.index(symbol)]

            # Determine new aspect boundaries
            new_from = int(opinion.attrib['from']) - 1 - 2 * symbols.index(symbol)
            new_to = int(opinion.attrib['to']) - 1 - 2 * symbols.index(symbol)
            opinion.attrib['from'] = new_from.__str__()
            opinion.attrib['to'] = new_to.__str__()

            # Remove symbols
            sentence_text = sentence_text.replace(symbol[0], '', 1)
            sentence_text = sentence_text.replace([1], '', 1)

        sentence.find(".//text").text = sentence_text

    tree.write(filename)
    return tree
