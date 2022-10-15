from sentence_splitter import split_text_into_sentences


def get_first_sentence(text: str) -> str:
    try:
        return split_text_into_sentences(text, language="en")[0]
    except IndexError:
        return text
