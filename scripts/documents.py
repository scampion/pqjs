import json
import sys

import requests


def get_documents(nb_of_docs):
    headers = {
        'Accept': '*/*',
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
        }

    json_data = {"dateCriteria": {"field": "DATE_DOCU", "startDate": None, "endDate": None}, "years": [], "terms": [],
                 "accesses": [],
                 "types": ["ATAG", "IDAN", "BRIE", "STUD", "EURCATALOG", "EURDIVERS", "EUROTH", "ETFICHTEC", "PUBLI",
                           "ARATD"],
                 "authorCodes": [], "fragments": [], "geographicalAreas": [], "eurovocs": [],
                 "directoryCodes": [], "subjectHeadings": [], "policyAreas": [], "institutions": [],
                 "authorAuthorities": [], "recipientAuthorities": [], "authorOrRecipientAuthorities": [], "nbRows": nb_of_docs,
                 "references": [], "relations": [], "authors": [], "currentPage": 1, "sortAndOrder": "DATE_DOCU_DESC",
                 "excludesEmptyReferences": False, "fulltext": None, "title": None, "summary": None
                 }

    response = requests.post(
        'https://www.europarl.europa.eu/RegistreWeb/services/search',
        headers=headers,
        json=json_data
    )

    for r in response.json()['references']:
        doc = {'id': r['docId'],
               'date': r['dateDocu'],
               'reference': r['reference']}

        for f in r['fragments']:
            if f['value'] == 'FULL':
                version = f['versions'][0]
                doc['title'] = version['title']
                doc['summary'] = version['summary']
                doc['url'] = version['fileInfos'][0]['url']
        yield doc


if __name__ == '__main__':
    nb_of_docs = sys.argv[1]
    print(json.dumps(list(get_documents(nb_of_docs)), indent=2))
