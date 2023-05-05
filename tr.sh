#!/bin/sh
#BSUB -J hvsmr
#BSUB -e %J.err 
#BSUB -o %J.out 
#BSUB -q gpu 
#BSUB –R “select [ngpus>0] rusage [ngpus_excl_p=1]
python train.py
