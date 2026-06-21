from pathlib import Path
import fitz
from spacy.lang.en import English
import re

""

#size of chunk
SENTENCE_CHUNK_SIZE = 10

#minimum token len for valid chunk. to filter low sentence / low informational data
MIN_TOKEN_LEN = 30


class Document:
    """
    contains informaiton and handles parsing of document
    """
    def __init__(self, path, type):
        """
        initiallizes a new documewnt
        :param path: path to document
        :param type: type of document
        """
        self.path = path
        self.type = type
        self.pages = []

        self.parse_document()

    def parse_document(self):
        """
        general document parser
        """
        if self.type == 'pdf':
            self.parse_pdf()

        self.parse_sentences()

    def parse_pdf(self):
        """
        parses pdf files
        """
        with fitz.open(self.path) as f:
            for page_num, page in enumerate(f):
                raw_text = page.get_text()
                document_text = self.text_format(raw_text)
                current_page = Page(document_text, page_num)
                self.pages.append(current_page)


    def parse_sentences(self):
        """
        parses for sentences inside a document using an NLP engine
        """
        nlp = English()
        nlp.add_pipe("sentencizer")
        for page in self.pages:
            sentences = list(nlp(page.contents).sents)
            page.page_sentences = [str(sentence) for sentence in sentences]
            page.page_setence_count = len(page.page_sentences)
            page.page_sentence_chunks = split_sentences(page.page_sentences)


    def text_format(self, text):
        """
        format informational text
        :param text: text to format
        :return: formatted text
        """
        text = text.replace('\n', ' ').strip()
        return text

class Page:
    """
    A page inside a document, contains information about the page and the page data itself
    """
    def __init__(self, text, page_num=0):
        """
        initiallizes a new page
        :param text: data inside the page
        :param page_num: page number
        """
        self.page_num = page_num
        self.contents = text
        self.page_char_count = len(self.contents)
        self.page_word_count = len(self.contents.split(' '))
        self.page_estimated_token_count = self.page_char_count // 4
        self.page_sentences = []
        self.page_setence_count = 0
        self.page_sentence_chunks = []

class Chunk:
    """
    data chunks derived from document
    """
    def __init__(self, chunk):
        """
        initiallizes a new chunk
        :param chunk: chunk of data
        """
        self.contents = self.join_and_format_chunk(chunk)
        self.chunk_char_count = len(self.contents)
        self.chunk_word_count = len([i for i in self.contents.split(' ')])
        self.chunk_estimated_token_count = len(self.contents) // 4
        self.embedding = []

    def join_and_format_chunk(self, chunk):
        """
        converts chunks into correct format
        :param chunk: chunk data
        :return: formattd chunk
        """
        joined_chunk = ''.join(chunk).replace('  ', ' ').strip()
        joined_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_chunk)
        return joined_chunk


def parse_chunks(document):
    """
    parses chunks
    :param document: document to parse
    :return: list of chunks
    """
    pages_chunks = []
    for page in document.pages:
        for sentence_chunk in page.page_sentence_chunks:
            chunk_dict = {}
            chunk_dict['page_num'] = page.page_num
            current_chunk = Chunk(sentence_chunk)
            chunk_dict['chunk'] = current_chunk
            pages_chunks.append(chunk_dict)

    return filter_chunks(pages_chunks)

def filter_chunks(chunk_list):
    """
    filters low quality/short chunks
    :param chunk_list: chunks list
    :return: filtered chunks list
    """
    to_delete = []
    for chunk_data in chunk_list:
        if chunk_data['chunk'].chunk_estimated_token_count < MIN_TOKEN_LEN:
            to_delete.append(chunk_data)
    for chunk in to_delete:
        chunk_list.remove(chunk)

    return chunk_list


def split_sentences(sentences):
    chunks = []
    for i in range(0, len(sentences), SENTENCE_CHUNK_SIZE):
        chunks.append(sentences[i:i + SENTENCE_CHUNK_SIZE])
    return chunks

