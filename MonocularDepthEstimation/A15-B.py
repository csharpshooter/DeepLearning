# i_path = r'D:\Development\TSAI\EVA\MaskRCNN Dataset\OverLayedImages\batch_1'
# paths = utils.Utils.get_all_file_paths(i_path)
# images = []
# for path in paths[0]:
#     images.append(np.array(Image.open(path)))
#
# images = np.array(images)
# mean, std = utils.Utils.calculate_mean_std_deviation(images)

# [0.4222796, 0.44544333, 0.44153902]
# [0.28497052, 0.24810323, 0.2657039]
# Monocular Std and Mean

from multiprocessing import freeze_support

from src.dataset.monocularhelper import MonocularHelper
from src.imports import *


def main():
    helper = MonocularHelper()

    # TODO
    # path = helper.download_dataset(folder_path="data")

    # TODO
    # dict = helper.get_id_dictionary(path=path)

    # TODO
    # values, classes = helper.get_class_to_id_dict(id_dict=dict, path=path)

    # final_output = r'/media/abhijit/DATA/Development/TSAI/EVA/MaskRCNN Dataset/OverLayedImages'
    # final_output_mask = r'/media/abhijit/DATA/Development/TSAI/EVA/MaskRCNN Dataset/OverLayedMask'
    # final_output_dm = r'/media/abhijit/DATA/Development/TSAI/EVA/MaskRCNN Dataset/OverLayedDepthMasks'
    # bg_path = r'/media/abhijit/DATA/Development/TSAI/EVA/MaskRCNN Dataset/Background'

    # final_output = r'/home/abhijit/EVARepo/MonocularDS/OverLayedImages'
    # final_output_mask = r'/home/abhijit/EVARepo/MonocularDS/OverLayedMask'
    # final_output_dm = r'/home/abhijit/EVARepo/MonocularDS/OverLayedDepthMasks'
    # bg_path = r'/home/abhijit/EVARepo/MonocularDS/Background'

    final_output = r'C:\MonocularDS\OverLayedImages'
    final_output_mask = r'C:\MonocularDS\OverLayedMask'
    final_output_dm = r'C:\MonocularDS\OverLayedDepthMasks'
    bg_path = r'C:\MonocularDS\Background'
    # torch.backends.cudnn.benchmark = True

    train_data, train_label, test_data, test_label = helper.get_train_test_data(masks_folder=final_output_mask,
                                                                                images_folder=final_output,
                                                                                depth_masks_folder=final_output_dm,
                                                                                no_of_batches=40,
                                                                                total_images_count=400000,
                                                                                bg_folder=bg_path)

    print(len(train_label))
    print(len(test_label))

    batch_size = 16

    mean = [0.4222796, 0.44544333, 0.44153902]
    std = [0.28497052, 0.24810323, 0.2657039]

    train_transforms, test_transforms = preprochelper.PreprocHelper.getpytorchtransforms(mean, std, 64)
    ds = dst.Dataset()

    train_dataset = ds.get_monocular_train_dataset(train_image_data=train_data, train_image_labels=train_label,
                                                   train_transforms=train_transforms)

    test_dataset = ds.get_monocular_test_dataset(test_image_labels=test_label, test_image_data=test_data,
                                                 test_transforms=test_transforms)

    torch.manual_seed(1)

    dataloader = dl.Dataloader(traindataset=train_dataset, testdataset=test_dataset, batch_size=batch_size)
    train_loader = dataloader.gettraindataloader()
    test_loader = dataloader.gettestdataloader()

    import torch.nn as nn
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")

    # cnn_model, device = utils.Utils.createDepthModel()
    # optimizer = utils.Utils.createoptimizer(cnn_model, lr=0.5, momentum=0.9, weight_decay=1e-5)  # 1e-5

    cnn_model, device = utils.Utils.createMonocularModel()
    optimizer = utils.Utils.createoptimizer(cnn_model, lr=0.01, momentum=0.9, weight_decay=1e-5)  # 1e-5

    for name, param in cnn_model.named_parameters():
        #     print(name)
        #     print(param)
        if "bn1" in name or "bn2" in name or "double_conv" in name:
            i = 0
        #         nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
    #     elif "bias" in name:
    #         nn.init.constant_(param, 0)

    sample = next(iter(train_loader))

    imgs = sample[0][0]

    utils.Utils.show(imgs, nrow=4)

    train_model = train.TrainModel()

    train_model.showmodelsummary(model=cnn_model, input_size=[(4, 3, 64, 64)])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.05, patience=1,
                                                           verbose=True, threshold=0.01, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-08)

    lr_data = []
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    epochs = 5

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(cnn_model))

    # In[ ]:

    # from kornia.losses import SSIM
    # loss_fn = SSIM(window_size=3, reduction='mean')
    from torch.nn import BCEWithLogitsLoss, SmoothL1Loss, MSELoss, BCELoss
    # loss_fn = BCEWithLogitsLoss()
    # loss_fn = SmoothL1Loss()
    # loss_fn = MSELoss()
    from src.train.customlossfunction import DiceLoss
    loss_fn = DiceLoss()
    # loss_fn = BCELoss(reduction='mean')
    show_output = False
    infer_index = 2
    for epoch in range(1, epochs):
        print("EPOCH:", epoch)

        tr_out = train_model.train_Monocular(cnn_model, device, train_loader, optimizer, epoch, loss_fn, show_output,
                                             infer_index)
        ts_out, dice_loss = train_model.test_Monocular(cnn_model, device, test_loader, class_correct, class_total,
                                                       epoch, lr_data, loss_fn,
                                                       show_output, infer_index)

        from src.utils.utils import Utils

        Utils.show(tr_out.detach().cpu(), nrow=4)
        Utils.show(ts_out.detach().cpu(), nrow=4)

        scheduler.step(dice_loss)

    # train_losses, train_acc = train_model.gettraindata()
    # test_losses, test_acc = train_model.gettestdata()
    # utils.Utils.savemodel(model=cnn_model, epoch=epochs, path="savedmodels/finalmodelwithdata.pt",
    #                       optimizer_state_dict=optimizer.state_dict
    #                       , train_losses=train_losses, train_acc=train_acc, test_acc=test_acc,
    #                       test_losses=test_losses, lr_data=lr_data, class_correct=class_correct, class_total=class_total)


if __name__ == '__main__':
    freeze_support()
    main()
