import os


class LogRec:
    def __init__(self):
        return

    def set(self, train, epoch, iter_num, loss=None, lr=None, src=None, mse=None):
        self.train = train
        self.epoch = epoch
        self.iter_num = iter_num
        self.loss = loss
        self.src = src
        self.lr = lr
        self.mse = mse

    def as_csv(self):
        msg_list = []
        msg_list.append('train' if self.train else 'val')
        msg_list.append(str(self.epoch))
        msg_list.append(str(self.iter_num))
        msg_list.append(str(self.loss))
        msg_list.append(str(self.src))
        msg_list.append(str(self.lr))
        msg_list.append(str(self.mse))
        return ",".join(msg_list)

class Logger:
    def __init__(self):
        self.filename = None
        self.f = None
        return

    def open(self, filename):
        dirs, file = os.path.split(filename)
        os.makedirs(dirs, exist_ok=True)
        self.filename = filename
        self.f=open(filename, 'wt')


    def write(self, train, epoch, epoch_samples=None, sample=None, loss=None, lr=None, src=None, mse=None, **msg):

        if self.f is None:
            return

        iter_num = None
        if train:
            iter_num = epoch * epoch_samples + sample

        rec = LogRec()
        rec.set(train, epoch, iter_num, loss, lr, src, mse)
        csv_rec = rec.as_csv()
        self.f.write(csv_rec+'\n')
        self.f.flush()

        return
