me of the job --  
#BSUB -J swin_bra_test 
# -- specify queue – 
#BSUB -q normal 
# -- number of processors -- 
#BSUB -n 40
# --specify that the cores MUST BE on a single host! -- 
#BSUB -R "span[ptile=40]" 
# --Specify the output and error file. %J is the job-id -- 
# -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -oo %J.out 
#BSUB -eo %J.err 
python test.py

