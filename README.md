# legal-summary
Legal document summarizer using BERT and transformers from huggingface.co
Mostly a Proof-of-Concept, but fully working pipeline for OCR and text summaries.

# INSTALL
- on a mac: brew install tesseract and rust
- pip3 install pytesseract, pdf2image, nltk, transformers, yake, bert-extractive-summarizer
- if conda env : conda install pytorch scikit-learn numpy
On M1 Mac chip, its easier to install scikit-learn, numpy etc. with conda's pre-compiled packages!

# RTFM
To summarise legal docs. Or other docs. But works with legal docs as well,
which is a special case of difficult docs.

legal-summary traverses a folder for PDFs and OCRs them
then extracts all paragraphs and summarises them using e.g. Google's T5 GAN ML neural net
then summarises the concatenated summaries for an overview (higher quality than summarising
the entire document). Other ML nets can be used, see Hugging Face models.

We cannot use the usual statistical summaries methods like TextRank etc.
because a legal doc isn't such that more occurence of a word/sentence means its more
important... Equally, a single occurence of a word or sentence may be extremely important,
so we must record these as well.

We must rewrite each paragraph to a smaller paragraph with the same meaning, thus
using a GAN. 