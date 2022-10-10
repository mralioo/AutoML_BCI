#!/bin/bash

#$ -binding linear:4 # request 4 cpus (8 with Hyperthreading) (some recommend 4 per GPU)
#$ -N ex       # set consistent base name for output and error file (allows for easy deletion alias)
#$ -q all.q    # don't fill the qlogin queue (can some add why and when to use?)
#$ -cwd        # change working directory (to current)
#$ -V          # provide environment variables
##$ -t 1-100    # start 100 instances: from 1 to 100

#$ -M ali.alouane@outlook.de     # (debugging) send mail to address...
#$ -m ea                            # ... at job end and abort

##$ -t 1-100    # start 100 instances: from 1 to 100
##$ -o jobs_outputs/$JOB_NAME/$JOB_ID-$TASK_ID.o
##$ -e jobs_outputs/$JOB_NAME/$JOB_ID-$TASK_ID.e

#$ -o jobs_outputs/$JOB_NAME/$JOB_ID.o
#$ -e jobs_outputs/$JOB_NAME/$JOB_ID.e

##$ -l mem_free=500M
#echo "Only runs on nodes which have more than 500 megabytes of free memory"

# if you also want to request a GPU, add the following line to the above block:
#$ -l cuda=1  # request one GPU

echo "I am a job task with ID $JOB_ID"

python -m bbcpy.train.train_eegnet_v4_pipeline -f bbcpy/params/baselines_eegnet_v4.yaml