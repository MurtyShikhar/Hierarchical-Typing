import gzip, json, sys, spacy
from collections import defaultdict as ddict
from tqdm import tqdm
from spacy.tokens import Span
import glob

# load in unicode
reload(sys)
sys.setdefaultencoding("utf-8")


def get_spans(doc, mention_boundaries):
    sz = len(doc)
    mention_spans = []
    for i in xrange(sz):
        # start of mention
        if doc[i].idx in mention_boundaries:
            st = i
            j = i+1
            boundary_en, entity = mention_boundaries[doc[i].idx]
            while j < sz and doc[j].idx < boundary_en:
                j += 1

            mention_spans.append((Span(doc, i, j), entity))

    return mention_spans


def write_doc(wiki_article, spacy_doc, file):
    text = wiki_article[u'text']
    anchor_texts = wiki_article[u'anchorTexts']

    mention_boundaries = {}

    for anchor_text in anchor_texts:
        st = anchor_text[u'offset']
        en = anchor_text[u'offset'] + anchor_text[u'length']
        mention_boundaries[st] = (en, anchor_text[u'link'])


    spans = get_spans(spacy_doc, mention_boundaries)

    for span, entity in spans:
        curr_span_sent = span.sent
        curr_span_st = span.start
        curr_span_en = span.end
        curr_example = []
        for token in curr_span_sent:
            if token.i == curr_span_st:
                curr_example.append("<target>")
            if token.i == curr_span_en:
                curr_example.append("</target>")
            curr_example.append(token.text)


        curr_example = " ".join(curr_example)
        curr_example = curr_example.replace('\r', '').replace('\n', '').lstrip().rstrip()
        file.write("%s\t%s\n" %(entity, curr_example))


def parse_file(file_name, nlp):
    wiki_articles = []
    f = gzip.open(file_name)
    for line in f:
        wiki_articles.append(json.loads(line))
    f.close()
    f = open("%s.processed.gz" %file_name, "w")
    all_texts = (wiki_article[u'text'] for wiki_article in wiki_articles)

    #for i, doc in tqdm(enumerate(all_texts), total = len(wiki_articles)):
    for i, spacy_doc in tqdm(enumerate(nlp.pipe(all_texts, batch_size = 2000)), total = len(wiki_articles)):
        #spacy_doc = nlp(doc)
        wiki_article = wiki_articles[i]
        write_doc(wiki_article, spacy_doc, f)

    f.close()


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')
    file_name = sys.argv[1]

    files = glob.glob("%s/*.gz" %file_name)

    for file in tqdm(files, total = len(files)):
        parse_file(file, nlp)
