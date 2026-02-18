import os
os.environ["KAGGLEHUB_CACHE"] = os.getcwd()

import kagglehub

path = kagglehub.dataset_download("vpapenko/nails-segmentation")


print("Файлы скачаны в:", path)
