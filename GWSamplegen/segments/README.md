the segment files were fetched using the publicly available GWOSC timelines.

e.g. for Hanford O1 data:
curl https://gwosc.org/timeline/segments/O1/H1_DATA/1126051217/11203200/ > H1_O1.txt

The event GPS times were fetched from the GWTC catalogues on GWOSC, which are fetched as https://gwosc.org/eventapi/json/shortname
e.g. https://gwosc.org/eventapi/json/GWTC-3-confident/, https://gwosc.org/eventapi/json/GWTC-4.0