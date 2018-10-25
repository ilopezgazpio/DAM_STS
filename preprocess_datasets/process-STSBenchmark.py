import os
import sys
import argparse
import numpy as np
# from nltk.corpus import stopwords

punctuations = ['(','-lrb-','.',',','-','?','!',';','_',':','{','}','[','/',']','...','"','\'',')', '-rrb-']

def main(arguments):
    '''
    Read STSBenchmark official dataset and produce txt files for sent1, sent2 and similarity score
    '''
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_folder', help="location of folder with the sick files")
    parser.add_argument('--out_folder', help="location of the output folder")
    
    args = parser.parse_args(arguments)
    
    for split in ["train", "dev", "test"]:        
        src_out = open(os.path.join(args.out_folder, "src-"+split+".txt"), "w")
        targ_out = open(os.path.join(args.out_folder, "targ-"+split+".txt"), "w")
        label_out = open(os.path.join(args.out_folder, "label-"+split+".txt"), "w")

        for line in open(os.path.join(args.data_folder, "sts-"+split+".csv"),"r"):
            d = line.split("\t")
            label = d[4].strip().lower()
            premise = d[5].strip().lower()
            hypothesis = d[6].strip().lower()
            
            for punct in punctuations:
                if punct in premise:
                    premise = premise.replace(punct,"")
            #premise = [w for w in premise.split() if w not in stopwords]
            #premise = " ".join(premise)
            
            for punct in punctuations:
                if punct in hypothesis:
                    hypothesis = hypothesis.replace(punct,"")
            #hypothesis = [w for w in hypothesis.split() if w not in stopwords]
            #hypothesis = " ".join(hypothesis)

            src_out.write(premise + "\n")
            targ_out.write(hypothesis + "\n")
            label_out.write(label + "\n")

        src_out.close()
        targ_out.close()
        label_out.close()
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
