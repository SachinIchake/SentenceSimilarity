with open('/home/atom/UST/SentenceSimilarity/sampleData.txt') as fc :
    lines = fc.readlines()
    for i,line in enumerate(lines):
        if i%2==0:
            print(line.strip() +'\t' + 'answer_'+str(i))
