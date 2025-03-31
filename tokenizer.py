import re
from nltk.tokenize import sent_tokenize, TreebankWordTokenizer

ABBREVIATIONS = {"Mr.", "Mrs.", "Dr.", "Ms.", "Prof.", "Sr.", "Jr.", "vs.", "etc."}

def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "URL", text)
    text = re.sub(r"@\w+", "MENTION", text)
    text = re.sub(r"#\w+", "HASHTAG", text)
    text = re.sub(r"\b\d+(\.\d+)?\s?%", "PERCENTAGE", text)
    text = re.sub(r"\b\d+\s?(years old|yrs old|yo|years|yrs)\b", "AGE", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d{1,2}:\d{2}(?:\s?[APap][Mm])?\b", "TIME", text)
    text = re.sub(r"\b\d+\s?(seconds|minutes|hours|days|weeks|months|years)\b", 
                  "TIME_PERIOD", text, flags=re.IGNORECASE)
    return text

def clean_text(text):
    paragraphs = text.split("\n\n")
    cleaned_paragraphs = []

    for paragraph in paragraphs:
        cleaned_paragraph = re.sub(r"\s*\n\s*", " ", paragraph)
        cleaned_paragraphs.append(cleaned_paragraph.strip())

    return "\n\n".join(cleaned_paragraphs)

def fix_abbreviation_splits(text):
    for abbr in ABBREVIATIONS:
        text = re.sub(rf"\b{re.escape(abbr)}\s", abbr.replace(".", "__TEMP__") + " ", text)
    
    return text

def restore_abbreviations(text):
    return text.replace("__TEMP__", ".")

def custom_nlp_tokenizer(text):
    text = preprocess_text(text)
    text = clean_text(text)

    tokenizer = TreebankWordTokenizer()

    text = fix_abbreviation_splits(text)

    sentences = sent_tokenize(text)

    sentences = [restore_abbreviations(sent) for sent in sentences]

    tokenized_sentences = []
    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        tokens = ['<s>'] + tokens + ['</s>']
        tokenized_sentences.append(tokens)

    return tokenized_sentences

if __name__ == '__main__':
    text = input("Enter text: ")
    result = custom_nlp_tokenizer(text)
    print("Tokenized text:", result)