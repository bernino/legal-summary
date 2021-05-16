import pytesseract as pt
import pdf2image
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
# from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
import os
import yake
# from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModel
from summarizer import Summarizer,TransformerSummarizer
from transformers import *

# from https://theaidigest.in/summarize-text-document-using-transformers-and-bert/

# First, some setup
# Instantiating the model and tokenizer with gpt-2
# tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
# model=GPT2LMHeadModel.from_pretrained('gpt2')

# Instantiating the model and tokenizer with Google's T5
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# model_name = 't5-base'
# tokenizer_t5 = T5Tokenizer.from_pretrained('t5-small')

# This uses the bert-large-uncased model
# bert_legal_model = Summarizer()
# result = model(text, min_length=60, ratio=0.01)
# print(result)

# Load model, model config and tokenizer via Transformers
# Change model as you see fit, see huggingface.co
# model_name = 'distilbert-base-uncased'
# model_name = 'saibo/legal-roberta-base'
model_name = 'nlpaueb/legal-bert-base-uncased'
# model_name = 'nsi319/legal-pegasus'

# The setup of huggingface.co
custom_config = AutoConfig.from_pretrained(model_name)
custom_config.output_hidden_states=True
custom_tokenizer = AutoTokenizer.from_pretrained(model_name)
custom_model = AutoModel.from_pretrained(model_name, config=custom_config)
bert_legal_model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)
print('Using model {}\n'.format(model_name))

# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# padding = "max_length" 

# YAKE
# Yet Another Keyword Extractor (Yake) library selects the most important keywords using 
# the text statistical features method from the article. With the help of YAKE, you can 
# control the extracted keyword word count and other features.
# YAKE keyword extractor settings
kw_extractor = yake.KeywordExtractor()
language = "en"
max_ngram_size = 1
deduplication_threshold = 0.9
numOfKeywords = 10
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)

# nltk settings
# nltk automatically checks if already downloaded so don't worry
nltk.download('punkt')

# Then the app...
#
# Turn the PDF into images
# We do not want images to be too big, dpi=300?
# All our images should have the same size (depends on dpi), width=1654 and height=2340
# these settings leads to an image at about 500kB
# TODO: keep tabs on how many pages in total have been processed
path = os.getcwd()
folder_name = 'pdfs'
path = os.path.join(path, folder_name) 

list_of_files = []
for root, dirs, files in os.walk(path):
    for file in files:
        if(file.endswith(".pdf")):
            # print(os.path.join(root,file))
            list_of_files.append(os.path.join(root,file))

print("\nProcessing {} files...\n".format(len(list_of_files)))
total_pages = 0

for filename in list_of_files:
    print(filename)
    file = os.path.splitext(os.path.basename(filename))[0]
    pages = pdf2image.convert_from_path(pdf_path=filename, dpi=400, size=(1654,2340))
    total_pages += len(pages)
    print("\nProcessing the next {} pages...\n".format(len(pages)))

    # Then save all pages as images and convert them to text except the last page
    # TODO: create this as a function
    content = ""
    dir_name = 'images/' + file + '/' 
    os.makedirs(dir_name, exist_ok=True)
    # If folder doesn't exist, then create it.
    for i in range(len(pages)-1):
        pages[i].save(dir_name + str(i) + '.jpg')
        # OCR the image using Google's tesseract
        content += pt.image_to_string(pages[i])

    # 'content' is now a large set of paragraphs for each PDF file, let's loop over them and summarise
    # with the T5 model. A paragraph is assumed \n\n which is obviously wrong
    # TODO: use nltk TextTilingTokenizer (?) for cleaner paragraph detection? and clean up
    # using stopwords etc.?
    # An alternative is to do each sentence and summarise them into mini sentences... but
    # that would probably discard some of the context the GAN needs?
    # enumerating just in case we need it
    summary_text = ""
    for i, paragraph in enumerate(content.split("\n\n")):
        # use NLTK to prettify and detect sentences
        # paragraph = str(sent_tokenize(paragraph))
        # get rid of intra newlines and tabs
        # get rid of empty paragraphs and one word paras and extra whitespaces
        paragraph = paragraph.replace('\n',' ')
        paragraph = paragraph.replace('\t','')
        paragraph = ' '.join(paragraph.split())
        # count words in the paragraph and exclude if less than 4 words
        tokens = word_tokenize(paragraph)
        # only do real words
        tokens = [word for word in tokens if word.isalpha()]
        # print("\nTokens: {}\n".format(len(tokens)))
        # only do sentences with more than 1 words excl. alpha crap
        if len(tokens) <= 1:
            continue
        # Perhaps also ignore paragraphs with no sentence?
        sentences = sent_tokenize(paragraph)
        # print("\nSentences: {}\n".format(len(sentences)))
        # if len(sentences) == 0:
        #     continue

        # recreate paragraph from the only words tokens list
        paragraph = ' '.join(tokens)

        print("\nParagraph:")
        print(paragraph+"\n")
        # T5 needs to have 'summarize' in order to work:
        # text = "summarize:" + paragraph
        text = paragraph
        # encoding the input text
        # input_ids=tokenizer_t5.encode(text, return_tensors='pt')
        # input_ids=tokenizer_t5.encode(text, return_tensors='pt', max_length=512)
        # Generating summary ids
        # TODO: understand hyperparameters incl. early_stopping
        # hyperparameters inspired by sentence length https://prowritingaid.com/Sentence-Length-Average
        # summary_ids = t5_model.generate(input_ids,
        #                                     num_beams=4,
        #                                     no_repeat_ngram_size=2,
        #                                     min_length=1,
        #                                     max_length=30,
        #                                     early_stopping=False)

        # Decoding the tensor and printing the summary
        # t5_summary = tokenizer_t5.decode(summary_ids[0], skip_special_tokens=True)
        # input_tokenized = custom_tokenizer.encode(text, return_tensors='pt',padding=padding,pad_to_max_length=True, max_length=6144,truncation=True)
        # summary_ids = model.generate(input_tokenized,
        #                                 num_beams=4,
        #                                 no_repeat_ngram_size=3,
        #                                 length_penalty=2,
        #                                 min_length=10,
        #                                 max_length=500)
        # summary = [custom_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
        summary = bert_legal_model(text, ratio = 0.1, min_length=10, max_length = 300)
        # summary = tokenizer_t5.decode(summary_ids[0], skip_special_tokens=True)
        summary_text += str(summary) + "\n\n"
        print("Summary:")
        print(summary)

    # Summary of concatenated summaries
    # TODO: clean text for \n and stop words?
    # text = "summarize:" + t5_text
    # input_ids=tokenizer_t5.encode(text, return_tensors='pt')
    # Generating summary ids
    # summary_ids = t5_model.generate(input_ids,
    #                                     num_beams=4,
    #                                     no_repeat_ngram_size=2,
    #                                     min_length=20,
    #                                     max_length=1000,
    #                                     early_stopping=False)

    # t5_summary = tokenizer_t5.decode(summary_ids[0], skip_special_tokens=True)
    summary = bert_legal_model(content, ratio=0.1)
    # input_tokenized = custom_tokenizer.encode(content, return_tensors='pt',padding=padding,pad_to_max_length=True, max_length=6144,truncation=True)
    # summary_ids = model.generate(input_tokenized,
    #                                 num_beams=4,
    #                                 no_repeat_ngram_size=3,
    #                                 length_penalty=2,
    #                                 min_length=150,
    #                                 max_length=500)
    # summary = [custom_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]

    # print("\nT5 complete summary:")
    # print(t5_summary)
    # print("\nT5 detailed summary:")
    # print(t5_text)

    # Extract keywords from all content
    # TODO: clean result for things such as agreement, Andrew, Martin, ...
    keywords = custom_kw_extractor.extract_keywords(content)
    keyword_list = ""
    print("\nKeywords:")
    for kw in keywords:
        keyword_list += str(kw[0]).lower() + " with prob " + str(kw[1]) + "\n"
    print(keyword_list)

    # write all to file for inspection
    all_text = "-------- The Keywords --------\n" + str(keyword_list) + "\n\n\n" \
        + "-------- The Summary --------\n" + str(summary) + "\n\n\n" \
        + "-------- The Larger Summary --------\n" + str(summary_text) + "\n\n\n" \
        + "-------- The Original Content --------\n" + str(content)

    with open('summaries/'+file+'-summary.txt', 'w') as f:
        f.write(all_text)
    # TODO: extract topic clusters see https://towardsdatascience.com/nlp-for-topic-modeling-summarization-of-legal-documents-8c89393b1534
    # TODO: word2vec comparing documents and graphing them
    # TODO: store summaries and topic clusters in a db with a link to the orig doc