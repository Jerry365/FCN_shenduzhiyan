import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import CamvidDataset
from evalution_segmentation import eval_semantic_segmentation
from FCN import FCN
import cfg
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt


def evaluate(model, val_data):
    net = model.eval()
    eval_loss = 0
    eval_acc = 0
    eval_miou = 0
    eval_class_acc = 0

    # 记录当前时间
    process_now = datetime.now()

    for i, sample in enumerate(val_data):
        val_img = Variable(sample['img'].to(device))
        val_label = Variable(sample['label'].to(device))

        out = net(val_img)
        out = F.log_softmax(out, dim=1)

        loss = criterion(out, val_label)

        # 损失累加
        eval_loss += loss.item()

        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        pre_label = [i for i in pre_label]

        val_label = val_label.data.cpu().numpy()
        val_label = [i for i in val_label]

        eval_metrics = eval_semantic_segmentation(pre_label, val_label)
        eval_acc += eval_metrics['mean_class_accuracy']
        eval_miou += eval_metrics['miou']

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - process_now).seconds, 3600)
    m, s = divmod(remainder, 60)

    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    val_str = ('Valid Loss: {:.5f} \t'
               'Valid Acc: {:.5f} \t'
               'Valid Mean IOU: {:.5f} \t'
               'Valid Class Acc : {:.5f}').\
        format(
               eval_loss/len(train_data),
               eval_acc/len(val_data),
               eval_miou/len(val_data),
               eval_class_acc/len(val_data)
            )

    print(time_str)
    print(val_str)

if __name__ == '__main__':

    ###################################  选择网络运行采用的设备  ###############################################
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("device = ", device)


    ###################################  载入训练集和验证集  ###################################################
    Cam_train = CamvidDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
    Cam_val = CamvidDataset([cfg.VAL_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)

    train_data = DataLoader(Cam_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=1)
    val_data = DataLoader(Cam_val, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=1)

    ####################################  创建网络实例对象并加载到上述选好的 device中去 ###########################
    net = FCN(12)
    net.to(device)

    ########################################### 选择要采用的优化器 和 损失函数并加载到 device 中去 ################
    criterion = torch.nn.NLLLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)


    ########################################### 统计网络中的参数量 #############################################
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("要训练的参数总量 :  ", pytorch_total_params)


    ########################################### 开辟空间存储权重参数 #############################################
    best = [0]

    agraphic_iou = plt.subplot(1, 1, 1)
    plt.ion()
    agraphic_iou.set_title('miou tendency')  # 添加子标题
    agraphic_iou.set_xlabel('epoch', fontsize=10)  # 添加轴标签
    agraphic_iou.set_ylabel('miou', fontsize=20)

    agraphic_miou = plt.subplot(1, 1, 1)
    plt.ion()
    agraphic_miou.set_title('miou tendency')  # 添加子标题
    agraphic_miou.set_xlabel('epoch', fontsize=10)  # 添加轴标签
    agraphic_miou.set_ylabel('miou', fontsize=20)

    agraphic_acc = plt.subplot(1, 1, 1)
    plt.ion()
    agraphic_acc.set_title('acc tendency')  # 添加子标题
    agraphic_acc.set_xlabel('epoch', fontsize=10)  # 添加轴标签
    agraphic_acc.set_ylabel('acc', fontsize=20)

    agraphic_class_acc = plt.subplot(1, 1, 1)
    plt.ion()
    agraphic_class_acc.set_title('class acc tendency')  # 添加子标题
    agraphic_class_acc.set_xlabel('epoch', fontsize=10)  # 添加轴标签
    agraphic_class_acc.set_ylabel('class_acc', fontsize=20)

    ######## 在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，###########################
    #####################################################导致结果不确定。如果设置初始化，则每次初始化都是固定的##########
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(4)
    torch.backends.cudnn.deterministic = True  # 保证每次运行结果一样

    ############################################### 网络开始训练 ################################################
    net = net.train()
    for epoch in range(cfg.EPOCH_NUMBER):
        ############################################### 显示训练进度 ################################################
        print("Epoch is {}/{}".format(epoch+1, cfg.EPOCH_NUMBER))

        ############################################### 学习率优化策略 ################################################
        if epoch % 50 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group["lr"] *= 0.5

        ############################################### 定义好好每一个历元的误差 ########################################
        train_loss = 0
        train_acc = 0
        train_miou = 0
        train_class_acc = 0

        ############################################### 载入数据和标签开始训练 ########################################
        for i, sample in enumerate(train_data):
            img = Variable(sample["img"].to(device))
            label = Variable(sample["label"].to(device))

            ############################################### 将图片载入网络中去 ########################################
            out = net(img)

            ############################################### 定义好好每一个历元的误差 ########################################
            out = F.log_softmax(out, dim=1)

            ########################################### 损失函数计算误差，梯度清空，反向传播，梯度更新 ##########################
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ########################################### 统计每一轮的误差#################### ##########################
            train_loss += loss.item()

            ########################################### 拿到网络预测结果图和标签进行评估指标计算 ##############################
            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = label.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metric = eval_semantic_segmentation(pre_label, true_label)

            ######################## 将每个 batch 的评估指标依次累加，统计完一个完整的 epoch ###################################
            train_acc += eval_metric["mean_class_accuracy"]
            train_miou += eval_metric["miou"]
            train_class_acc += eval_metric["class_accuracy"]

            ## 输出当前的 batch 占据全体数据的份数, len(train_data): 表示训练集图片总数 / batch_size, loss.item 表示当前 epoch 的误差##
            print('batch {}/{}  batch_loss {:.8f}'.format(i + 1, len(train_data), loss.item()))



        ######################################## 当前 epoch 的结果输出【训练集】 ############################################
        metric_description = 'Train Acc : {:.5f}\t Train Mean : {:.5f}\n Train class acc : {:}'.format(
            train_acc / len(train_data),
            train_miou / len(train_data),
            train_class_acc / len(train_data)
        )
        print(metric_description)

        plt.plot(epoch+1, train_acc / len(train_data), 'ro', label = 'Train acc')
        plt.plot(epoch + 1, train_miou / len(train_data), 'ro', label='Train miou')
        plt.plot(epoch + 1, train_class_acc / len(train_data), 'ro', label='Train class miou')


        ###########################################按照约定的 epoch 的结果输出【测试集】#######################################
        if epoch % 2 == 0:
            evaluate(net, val_data)

        ######################################## 依据训练结果的 miou 的大小进行模型权重保存 ###################################
        if max(best) <= train_miou / len(train_data):
            currentBest = train_miou / len(train_data)
            best.append(currentBest)
            torch.save(net.state_dict(), 'train_moiu = {:.5f}_{}.pth'.format(currentBest, epoch))

    plt.ioff()  # 关闭画图的窗口，即关闭交互模式s
    plt.savefig("{}data_.jpg".format("结果图"))