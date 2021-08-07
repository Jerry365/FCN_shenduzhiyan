BATCH_SIZE = 1
EPOCH_NUMBER = 2
TRAIN_ROOT = './CamVid/train'
TRAIN_LABEL = './CamVid/train_labels'
VAL_ROOT = './CamVid/val'
VAL_LABEL = './CamVid/val_labels'
TEST_ROOT = './CamVid/test'
TEST_LABEL = './CamVid/test_labels'
class_dict_path = 'CamVid/num_classes.csv'

# 352是为了整除
crop_size = (352, 480)
