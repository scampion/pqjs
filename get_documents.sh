#!/bin/bash 

# Query the publication service

curl 'https://www.europarl.europa.eu/RegistreWeb/services/search' \
  -H 'Accept: */*' \
  -H 'Content-Type: application/json' \
  -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 
Safari/537.36' \
'{"dateCriteria":{"field":"DATE_DOCU","startDate":null,"endDate":null},"years":[],"terms":[],"accesses":[],"types":["ATAG","IDAN","BRIE","STUD","EURCATALOG","EURDIVERS","EUROTH","ETFICHTEC","PUBLI","ARATD"],"authorCodes":[],"fragments":[],"geographicalAreas":[],"eurovocs":[],"directoryCodes":[],"subjectHeadings":[],"policyAreas":[],"institutions":[],"authorAuthorities":[],"recipientAuthorities":[],"authorOrRecipientAuthorities":[],"nbRows":10000,"references":[],"relations":[],"authors":[],"currentPage":1,"sortAndOrder":"DATE_DOCU_DESC","excludesEmptyReferences":false,"fulltext":null,"title":null,"summary":null}' 
\
  --compressed > results.json

# Download English PDF
cat results.json | jq | grep url  |grep EN.pdf |  sed 's/^.*http/http/g' | sed 's/".*$//'| xargs wget

# Build filenames list
cat results.json | jq | grep url  |grep EN.pdf |  sed 's/^.*http/http/g' | sed 's/".*$//'  | cut -d'/' -f 9 | sort -u > filenames.txt

# After index
cat  index.json| jq .filename | sort -u | sed s/\"//g > documents.txt
