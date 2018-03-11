from zipfile import ZipFile
import os

files = os.listdir('save')

myzip = ZipFile('saves.zip', 'w')
for file in files:
	file = os.path.join('save', file)
	myzip.write(file)

myzip.close()
