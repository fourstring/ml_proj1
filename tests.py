from model import MobileNaiveBayes
from splitter import FileDataSplitter

if __name__ == '__main__':
    ds = FileDataSplitter(source='price.csv')
    nb = MobileNaiveBayes('price_nb', ds)
    nb.prepare()
    nb.train()
    print(nb.report())
