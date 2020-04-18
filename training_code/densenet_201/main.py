from fastai.vision import *

torch.cuda.set_device(1)
path = '/mnt/RAID5/users/alfonsomedela/projects/kaggle_comp/Projects/mask_detection/data/train/'

tfms = get_transforms()
data = (ImageList.from_folder(path)
        .split_by_rand_pct(0.10, seed=666)
        .label_from_folder()
        .transform(tfms, size=512, padding_mode='reflection')
        .databunch(num_workers=4, bs=8)
        .normalize(imagenet_stats)
        )


if __name__ == '__main__':

    # Activate mixup
    MIXUP = True
    lr_find = True

    print(data)
    print(data.classes)

    if MIXUP:
        learn = cnn_learner(data, models.densenet201, metrics=[accuracy]).mixup()
        learn.model_dir = '/mnt/RAID5/users/alfonsomedela/projects/kaggle_comp/Projects/mask_detection/densenet_201/new_split/weights/'

        if lr_find:
            learn.lr_find()
            fig = learn.recorder.plot(return_fig=True)
            fig.savefig('lr_figure_freezed_mixup.png')

        else:
            learn.fit_one_cycle(40, slice(1e-3))
            learn.save('stage1_weights_mixup')

            learn.unfreeze()
            learn.fit_one_cycle(10, slice(1e-6, 1e-4))
            learn.save('stage2_weights_mixup')


    else:
        learn = cnn_learner(data, models.densenet201, metrics=[accuracy])
        learn.model_dir = '/mnt/RAID5/users/alfonsomedela/projects/kaggle_comp/Projects/mask_detection/densenet_201/new_split/weights/'


        learn.load('stage2_weights_mixup')
        learn.unfreeze()
        learn.fit_one_cycle(3, slice(1e-4))
        learn.save('stage3_weights')

        learn.unfreeze()
        learn.fit_one_cycle(5, slice(1e-5))
        learn.save('stage4_weights')

        learn.export()











