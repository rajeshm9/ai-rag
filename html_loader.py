from langchain.document_loaders import UnstructuredHTMLLoader
import requests

import bs4
from langchain_community.document_loaders import WebBaseLoader

def load_html_docs():
    """
    Loads and returns documents from an HTML file using UnstructuredHTMLLoader.
    """
    urls = [
        "https://www.3gpp.org/technologies/5g-system-overview",
        "https://fight.mitre.org/"
    ]

    docs = []
    for url in urls:
        html_content = requests.get(url).text
        with open("temp.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        loader = UnstructuredHTMLLoader("temp.html")            
        docs.extend(loader.load())
    return docs

def load_web_docs():
    """
    Loads and returns documents from a web page using WebBaseLoader.
    """
    urls = [
        "https://www.3gpp.org/technologies/5g-system-overview",
        "https://fight.mitre.org/"
    ]
    
    loader = WebBaseLoader(urls)
    return loader.load()