import os
import tarfile
from six.moves import urllib

download_url = 'https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz'
housing_path = '../datasets/housing'

def fetch_housing_data(path=housing_path, url=download_url):
    if os.path.exists(path) == False:
        os.mkdir(path)

    tgz_path = os.path.join(path, 'housing.tgz')
    urllib.request.urlretrieve(url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path)
    housing_tgz.close()

if __name__ == '__main__':
    fetch_housing_data()
