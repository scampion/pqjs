import json
import sys

import requests
from joblib import Memory

memory = Memory("cache", verbose=0)


@memory.cache
def get(params):
    headers = {
        'Accept': '*/*',
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
    }

    response = requests.post(
        'https://www.europarl.europa.eu/RegistreWeb/services/search',
        headers=headers,
        json=params
    )
    return response


def get_documents(nb_of_docs):
    json_data = {"dateCriteria": {"field": "DATE_DOCU", "startDate": None, "endDate": None}, "years": [], "terms": [],
                 "accesses": [],
                 "types": ["ATAG", "IDAN", "BRIE", "STUD", "EURCATALOG", "EURDIVERS", "EUROTH", "ETFICHTEC", "PUBLI",
                           "ARATD"],
                 "authorCodes": [], "fragments": [], "geographicalAreas": [], "eurovocs": [],
                 "directoryCodes": [], "subjectHeadings": [], "policyAreas": [], "institutions": [],
                 "authorAuthorities": [], "recipientAuthorities": [], "authorOrRecipientAuthorities": [],
                 "nbRows": nb_of_docs,
                 "references": [], "relations": [], "authors": [], "currentPage": 1, "sortAndOrder": "DATE_DOCU_DESC",
                 "excludesEmptyReferences": False, "fulltext": None, "title": None, "summary": None
                 }
    response = get(json_data)
    with open('response.json', 'w') as f:
        json.dump(response.json(), f, indent=2)

    for ref in response.json()['references']:
        for frag in ref['fragments']:
            if frag['value'] == 'FULL':
                for version in frag['versions']:
                    if version['language'] == 'EN':
                        for fi in version['fileInfos']:
                            if fi['typeDoc'] == 'application/pdf':
                                yield {'id': ref['docId'],
                                       'date': ref['dateDocu'],
                                       'reference': ref['reference'],
                                       'title': version['title'],
                                       'summary': version['summary'],
                                       'url': fi['url']}


if __name__ == '__main__':
    nb_of_docs = sys.argv[1]
    print(json.dumps(list(get_documents(nb_of_docs)), indent=2))
