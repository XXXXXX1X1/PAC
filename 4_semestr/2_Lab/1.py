import os
from glob import glob
import cv2
import random
DATA = r'C:\Users\salap\Desktop\Учеба\Pac\PAC\4_semestr\2_Lab\orl_faces'

persons = [person for person in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, person))]

by_person = {}
for person in persons:
    by_person[person] = sorted(glob(os.path.join(DATA, person, "*.pgm")))


def split_persons(by_person, seed = 42, train = 7, val = 2, test = 1):
    random.seed(seed)
    split = {"train": {}, "val": {}, "test": {}}
    for person, files in by_person.items():
        files = files[:]
        random.shuffle(files)

        split["train"][person] = files[:train]
        split["val"][person] = files[train:train+val]
        split["test"][person] = files[val:]
    return split

split = split_persons(by_person, seed=42)
print(len(split["train"]), len(split["val"]), len(split["test"]))

