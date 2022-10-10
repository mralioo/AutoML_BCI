#!/bin/bash

#$ -binding linear:4 # request 4 cpus (8 with Hyperthreading) (some recommend 4 per GPU)
#$ -N ex       # set consistent base name for output and error file (allows for easy deletion alias)
#$ -q all.q    # don't fill the qlogin queue (can some add why and when to use?)
#$ -cwd        # change working directory (to current)
#$ -V          # provide environment variables

#$ -N myjob              # output will go to  myjob.ojobid
#$ -l h_rt=10:00:00      # replace by your estimate for run time
#$ -l h_vmem=2G          # allocate 2GB of virtual memory
#$ -j yes                # stdout and stderr will go to  myjob.ojobid
#$ -cwd                  # start job in current working directory

#$ -o jobs_outputs/$JOB_NAME/$JOB_ID.o
#$ -e jobs_outputs/$JOB_NAME/$JOB_ID.e

##$ -M ali.alouane@outlook.de     # (debugging) send mail to address...
##$ -m ea                            # ... at job end and abort


cd /home/bbci/data/teaching/BCI-PJ2021SS/data

wget --content-disposition https://figshare.com/ndownloader/files/25265282
wget --content-disposition https://figshare.com/ndownloader/files/25265270
wget --content-disposition https://figshare.com/ndownloader/files/25265303
wget --content-disposition https://figshare.com/ndownloader/files/25265360
wget --content-disposition https://figshare.com/ndownloader/files/25265384
wget --content-disposition https://figshare.com/ndownloader/files/25265387
wget --content-disposition https://figshare.com/ndownloader/files/25265390
wget --content-disposition https://figshare.com/ndownloader/files/25265393
wget --content-disposition https://figshare.com/ndownloader/files/25265396
wget --content-disposition https://figshare.com/ndownloader/files/25265402
wget --content-disposition https://figshare.com/ndownloader/files/25265405
wget --content-disposition https://figshare.com/ndownloader/files/25265408
wget --content-disposition https://figshare.com/ndownloader/files/25265411
wget --content-disposition https://figshare.com/ndownloader/files/25265414
wget --content-disposition https://figshare.com/ndownloader/files/25265423
wget --content-disposition https://figshare.com/ndownloader/files/25265432
wget --content-disposition https://figshare.com/ndownloader/files/25265441
wget --content-disposition https://figshare.com/ndownloader/files/25265444
wget --content-disposition https://figshare.com/ndownloader/files/25265447
wget --content-disposition https://figshare.com/ndownloader/files/25265450
wget --content-disposition https://figshare.com/ndownloader/files/25265456
wget --content-disposition https://figshare.com/ndownloader/files/25265459
wget --content-disposition https://figshare.com/ndownloader/files/25265468
wget --content-disposition https://figshare.com/ndownloader/files/25265477
wget --content-disposition https://figshare.com/ndownloader/files/25265483
wget --content-disposition https://figshare.com/ndownloader/files/25265492
wget --content-disposition https://figshare.com/ndownloader/files/25265495
wget --content-disposition https://figshare.com/ndownloader/files/25265513
