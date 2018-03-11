from zipfile import ZipFile
import os

myzip = ZipFile('saves.zip', 'r')
files = os.listdir('save')
for file in files:
	file = os.path.join('save', file)
	os.remove(file)
myzip.extractall()
myzip.close()
