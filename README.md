to download word vectors, run
wget http://stanford.edu/~nkgarg/NYTembeddings/vectorsnyt2004-2007.txt

DO NOT COMMIT THIS TO GITHUB BECAUSE IT IS TOOOOOOOOOO LARGE

Files:
jobs.txt --A list of jobs (currently only contains single word)
countries.txt --A list of countries (currently only contains 10)
generate.py --Python script for generate users randomly
clustering.py --Visualization and perform clustering
select_embeddings.py --Given the list of word we need, get embeddings
                       and save them to a smaller file

To run the clustering:
./run.sh
