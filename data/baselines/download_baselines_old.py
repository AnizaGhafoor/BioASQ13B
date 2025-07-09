from urllib.request import urlopen
from lxml import etree
from pipemp import StepConverter, BaseProcess, Pipeline, Signals
import os
import json
import pubmed_parser as pp
import urllib.request
import argparse
import gzip
from lxml import etree

def parse_mesh_terms(medline):
    """
    A function to parse MESH terms from article
    Parameters
    ----------
    medline: Element
        The lxml node pointing to a medline document
    parse_subs: bool
        If True, parse mesh subterms as well.
    Returns
    -------
    mesh_terms: str
        String of semi-colon ``;`` spearated MeSH (Medical Subject Headings)
        terms contained in the document.
    """

    if medline.find("MeshHeadingList") is not None:
        mesh = medline.find("MeshHeadingList")
        mesh_terms_list = [
            m.find("DescriptorName").attrib.get("UI", "")
            + ":"
            + m.find("DescriptorName").text
            for m in mesh.getchildren()
        ]
        mesh_terms = "; ".join(mesh_terms_list)
    else:
        mesh_terms = ""
    return mesh_terms

def parse_pmid(medline):
    """
    A function to parse PMID from a given Pubmed Article tree
    Parameters
    ----------
    pubmed_article: Element
        The lxml node pointing to a medline document
    Returns
    -------
    pmid: str
        A string of PubMed ID parsed from a given
    """

    if medline.find("PMID") is not None:
        pmid = medline.find("PMID").text
        return pmid
    else:
        article_ids = medline.find("PubmedData/ArticleIdList")
        if article_ids is not None:
            pmid = article_ids.find('ArticleId[@IdType="pmid"]')
            if pmid is not None:
                if pmid.text is not None:
                    pmid = pmid.text.strip()
                else:
                    pmid = ""
            else:
                pmid = ""
        else:
            pmid = ""
    return pmid

def parse_article_info(
    medline, nlm_category=False, author_list=False
):
    """Parse article nodes from Medline dataset
    Parameters
    ----------
    pubmed_article: Element
        The lxml element pointing to a medline document
    year_info_only: bool
        see more details in date_extractor()
    nlm_category: bool
        see more details in parse_medline_xml()
    author_list: bool
        if True, return output as list, else
    reference_list: bool
        if True, parse reference list as an output
    parse_subs: bool
        if True, parse mesh terms with subterms
    Returns
    -------
    article: dict
        Dictionary containing information about the article, including
        `title`, `abstract`, `journal`, `authors`, `affiliations`, `pubdate`,
        `pmid`, `other_id`, `mesh_terms`, `pages`, `issue`, and `keywords`. The field
        `delete` is always `False` because this function parses
        articles that by definition are not deleted.
    """
    article = medline.find("Article")

    if article.find("ArticleTitle") is not None:
        title = pp.utils.stringify_children(article.find("ArticleTitle")).strip() or ""
    else:
        title = ""

    category = "NlmCategory" if nlm_category else "Label"
    if article.find("Abstract/AbstractText") is not None:
        # parsing structured abstract
        if len(article.findall("Abstract/AbstractText")) > 1:
            abstract_list = list()
            for abstract in article.findall("Abstract/AbstractText"):
                section = abstract.attrib.get(category, "")
                if section != "UNASSIGNED":
                    abstract_list.append("\n")
                    abstract_list.append(abstract.attrib.get(category, ""))
                section_text = pp.utils.stringify_children(abstract).strip()
                abstract_list.append(section_text)
            abstract = "\n".join(abstract_list).strip()
        else:
            abstract = (
                pp.utils.stringify_children(article.find("Abstract/AbstractText")).strip() or ""
            )
    elif article.find("Abstract") is not None:
        abstract = pp.utils.stringify_children(article.find("Abstract")).strip() or ""
    else:
        abstract = ""

    authors_dict = pp.medline_parser.parse_author_affiliation(medline)
    if not author_list:
        affiliations = ";".join(
            [
                author.get("affiliation", "")
                for author in authors_dict
                if author.get("affiliation", "") != ""
            ]
        )
        authors = ";".join(
            [
                author.get("lastname", "") + "|" + author.get("forename", "") + "|" +
                author.get("initials", "") + "|" + author.get("identifier", "")
                for author in authors_dict
            ]
        )
    else:
        authors = authors_dict
    journal = article.find("Journal")
    journal_name = " ".join(journal.xpath("Title/text()"))

    pmid = parse_pmid(medline)
    #doi = pp.medline_parser.parse_doi(medline)
    #references = pp.medline_parser.parse_references(medline, reference_list)
    mesh_terms = parse_mesh_terms(medline)
    keywords = pp.medline_parser.parse_keywords(medline)

    dict_out = {
        "title": title,
        "abstract": abstract,
        "journal": journal_name,
        "authors": authors,
        "pmid": pmid,
        "mesh_terms": mesh_terms,
        "keywords": keywords,
        "delete": False,
    }
    if not author_list:
        dict_out.update({"affiliations": affiliations})

    return dict_out

@StepConverter
class MP_PubmedLinks(BaseProcess):
    
    def __init__(self, baseline_year, **kwargs):
        super().__init__(**kwargs)
        self.baseline_year = baseline_year
        
    def __call__(self):
        
        url='https://lhncbc.nlm.nih.gov/ii/information/MBR/Baselines/'+str(self.baseline_year)+'.html'
        output=urlopen(url).read()
        tree=etree.fromstring(output.decode('utf-8'),etree.HTMLParser())
        
        for i in tree.xpath('/html/body/ul[2]/li/a'):
            if i.text.endswith("xml.gz"):
                yield i.get('href')

@StepConverter
class MP_PubmedDownloaderAndParser(BaseProcess):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __call__(self, queue_stream):
        
        for link in queue_stream:
            # download the file
            #print(link)
            with gzip.open(urlopen(link)) as xml_fd:
                for _, element in etree.iterparse(xml_fd, events=("end",)):
                    if element.tag == "MedlineCitation":
                        res = parse_article_info(
                            element
                        )
                        element.clear()
                        yield {k:res[k] for k in ["pmid", "title", "abstract", "mesh_terms", "keywords"]}

                
@StepConverter
class MP_JsonlWriter(BaseProcess):
    def __init__(self, jsonl_file, **kwargs):
        super().__init__(**kwargs)
        self.fOut = open(jsonl_file, "w")
        
    def __call__(self, queue_stream):
        
        for data in queue_stream:
            self.fOut.write(f"{json.dumps(data)}\n")
            yield Signals.SAMPLE_CONSUMED
            
        self.fOut.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("year")
    parser.add_argument("total_docs", type=int)
    args = parser.parse_args()
    
    pipeline = Pipeline([
        MP_PubmedLinks(args.year, num_processes=1),
        MP_PubmedDownloaderAndParser(num_processes=6, size_queue=1000),
        MP_JsonlWriter(f"baselines/pubmed_baseline_{args.year}.jsonl")
    ], total_samples=args.total_docs)
    
    pipeline.run(debug_inspect_queue_sizes=False)