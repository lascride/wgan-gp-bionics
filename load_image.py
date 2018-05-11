import numpy as np
import scipy.misc
import time

def make_generator(path, n_files, batch_size, dim=64, count=None):
    if count==None:
        epoch_count = [1]
    else: 
        epoch_count = [count]
    def get_epoch():
        images = np.zeros((batch_size, 3, dim, dim), dtype='int32')
        files = list(range(n_files))
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        epoch_count[0] += 1
        for n, i in enumerate(files):
            image = scipy.misc.imread("{}/{}.jpg".format(path, str(i)))
            images[n % batch_size] = image.transpose(2,0,1)
            if n > 0 and n % batch_size == 0:
                yield (images,)
    return get_epoch

def load(batch_size, data_dir='E:/project/project/image/input_3_64_10000_rot/10',dim=64,num=4096,count=None):
    return make_generator(data_dir, 4*num, batch_size,dim,count)

    

if __name__ == '__main__':
    train_gen = load(64)
    t0 = time.time()
    print('start')
    for i, batch in enumerate(train_gen(), start=1):
        print ("{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0]))
        if i == 1000:
            break
        t0 = time.time()