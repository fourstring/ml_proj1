from model import MobileLogisticRegression, MobileSVM
from splitter import FileDataSplitter

if __name__ == '__main__':
    ds = FileDataSplitter(source='price.csv')

    lr = MobileLogisticRegression('price_lr', 0.03, ds)
    lr.prepare()
    lr.train(valid_loss_threshold=0.1)
    lr.report()
    svm = MobileSVM('price_svm', ds)
    svm.prepare()
    svm.train()
    svm.report()
