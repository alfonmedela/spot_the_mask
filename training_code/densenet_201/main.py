from fastai.vision import *

# Choose cuda device number
torch.cuda.set_device(1)

#This is the path to the training data. The data should be divided in two folders: mask/no_mask
path = 'ROOT_PATH/train/'

tfms = get_transforms()
data = (ImageList.from_folder(path)
        .split_by_rand_pct(0.10, seed=666)
        .label_from_folder()
        .transform(tfms, size=512, padding_mode='reflection')
        .databunch(num_workers=4, bs=8)
        .normalize(imagenet_stats)
        )


if __name__ == '__main__':


    '''
    Steps to follow:
    1) Find lr. It is already done
    2) Run with lr_find=False and MIXUP=True
    3) Run with MIXUP=False'''

    # Activate mixup
    MIXUP = True
    lr_find = True

    print(data)
    print(data.classes)

    if MIXUP:
        learn = cnn_learner(data, models.densenet201, metrics=[accuracy]).mixup()
        # Set up model dir to save the weights
        learn.model_dir = 'ROOT_PATH/weights/'

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
        # Set up model dir to save the weights
        learn.model_dir = 'ROOT_PATH/weights/'

        learn.load('stage2_weights_mixup')
        learn.unfreeze()
        learn.fit_one_cycle(3, slice(1e-4))
        learn.save('stage3_weights')

        # learn.unfreeze()
        learn.fit_one_cycle(5, slice(1e-5))
        learn.save('stage4_weights')

        # Export final model. Make sure lowest validation loss is obtained. This was around 0.02-0.03 for seed=666, which gave me best result without ensembling
        learn.export()











