from os.path import join as pjoin
from os import listdir
import os
DATA_dir = pjoin('/scratch','yl148','data');
subjects = listdir(DATA_dir)
print(len(listdir(DATA_dir)))
f = open("tracking.sh", "w")
f.write('#!/bin/bash\n')
for subject in subjects:
  diffusion_path = pjoin(DATA_dir, subject, 'T1w', 'Diffusion')  
  subject_path = pjoin(DATA_dir, subject)  
  fiber_file = pjoin(diffusion_path, subject+'LocalTracking.npz')
  # print(fiber_file)
  if os.path.exists(fiber_file):
     print('File %s already exists, skipping...' % fiber_file)
     f.write('echo File %s already exists, skipping...\n' % fiber_file)
  else:
     f.write('sbatch %s.jb\n' %(subject))
  # f.write('rm %s.jb\n' %(subject))
  """
  #!/bin/bash
  #SBATCH --job-name=1000
  #SBATCH --ntasks=16
  #SBATCH --mem-per-cpu=32000m
  #SBATCH --time=15:00:00
  #SBATCH --mail-type=ALL
  srun /opt/apps/software/Core/Anaconda3/5.0.0/bin/python tracking.py | tee -a log.txt
  """
  with open('%s.jb' %(subject), 'w') as jb:
     jb.write('#!/bin/bash\n')
     jb.write('#SBATCH --job-name=%s\n'%(subject))
     jb.write('#SBATCH --mem-per-cpu=20000m\n')
     jb.write('#SBATCH --time=1:00:00\n')
     jb.write('srun /opt/apps/software/Core/Anaconda3/5.0.0/bin/python ../src/tracking.py %s\n' % subject_path)
     #jb.write('echo %s Done.' % subject)
for subject in subjects[:-1]:
    f.write('rm %s.jb\n' %(subject))
f.close() 
