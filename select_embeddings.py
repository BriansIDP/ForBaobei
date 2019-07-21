wordlist = []
with open('jobs.txt') as fin:
    for job in fin:
        wordlist.append(job.strip())

with open('countries.txt') as fin:
    for country in fin:
        wordlist.append(country.strip())

education = ['infant', 'primary', 'secondary', 'bachelor', 'master', 'doctoral', 'school', 'degree']
gender = ['male', 'female']
wordlist += education
wordlist += gender
print(wordlist)

embeddings = {}
with open('word_embeddings.txt') as fin:
    for i, line in enumerate(fin):
        elems = line.split()
        word = elems[0]
        vec = elems[1:]
        embeddings[word] = vec
        if i % 10000 == 0:
            print(i)

lines_to_write = []

for word in wordlist:
    vec = embeddings[word.lower()]
    lineelems = [word.lower()] + vec
    line = ' '.join(lineelems) + '\n'
    lines_to_write.append(line)

with open('used_word_embeddings.txt', 'w') as fout:
    fout.writelines(lines_to_write)
