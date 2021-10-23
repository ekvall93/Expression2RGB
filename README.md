# Expression data to RGB image 
## This is a light-weight implementation of RESEPT (https://github.com/OSU-BMBL/RESEPT).

## Installation
### python3.8

```console
terran@terran:~$ python
>>> from terran.AWS import AWSS3Diva
>>> awsDiva = AWSS3Diva(bucket="diva-data-storage")
>>> pid2emb = awsDiva.getEmbeddings(f"MAIN/1900_2021_KTH/")
>>> from terran.processMetaData import KMEANS
>>> K = KMEANS(pid2emb, 10, 1)
>>> awsDiva.uploadKMEANS(K, f"MAIN/1900_2021_KTH/","pkl")
```

### pip install -e .