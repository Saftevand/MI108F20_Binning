import os
from Bio import SeqIO
import gzip
import datetime
from pathlib import Path
import vamb_tools
import multiprocessing as _multiprocessing

'''
print(str(datetime.datetime.now()))
with gzip.open('C:/Users/M0107/Desktop/p10/Bin.gz', "rt") as handle:
    for seq_record in SeqIO.parse(handle, "fasta"):


f = open('t.fasta', 'w+')
lines = ['>RL|S1|C0\n','agtgact\n','>RL|S1|C6\n','gtgtgtg']
for x in lines:
    f.write(x)
f.close()



#with vamb_tools.Reader('C:/Users/M0107/Desktop/p10/Bin.gz', 'rb') as filehandle:
with vamb_tools.Reader('test.fasta', 'rb') as filehandle:
    #entries = vamb_tools.byte_iterfasta(filehandle)
    tnfs, contigname, lengths = vamb_tools.read_contigs(filehandle, minlength=4)
    print('test')
'''

if __name__ == '__main__':
    _multiprocessing.freeze_support()
    rpkms = vamb_tools.read_bamfiles(['C:/Users/M0107/Desktop/p10/vamb-master/test/data/one.bam',
                                      'C:/Users/M0107/Desktop/p10/vamb-master/test/data/two.bam',
                                      'C:/Users/M0107/Desktop/p10/vamb-master/test/data/three.bam'])
    print(str(rpkms))

    with vamb_tools.Reader('C:/Users/M0107/Desktop/p10/Bin.gz', 'rb') as filehandle:
        tnfs, contigname, lengths = vamb_tools.read_contigs(filehandle, minlength=4)





















