import os, shutil, json

if os.path.isdir('runs'):
    shutil.rmtree('runs')
    os.mkdir('runs')

with open('config/open_jobs.json', 'w') as f:
   reset = []
   json.dump(reset, f)