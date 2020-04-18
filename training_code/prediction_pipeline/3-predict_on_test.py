################################## alfonsomedela.com #################################################

import matplotlib
matplotlib.use('Agg')
from fastai.vision import *
import pandas as pd
import glob
import pickle

torch.cuda.set_device(2)

#load densenet201
learner_path = 'PATH_TO_WEIGHTS'
learn = load_learner(learner_path)

# Load on top prediction model
filename = 'PATH/confidence_final.sav'
confidence_model = pickle.load(open(filename, 'rb'))

def get_predictions(image_path):

    '''
    :param image_path: input path of the image here
    :return: predictions for the 5 tiles
    '''

    face = PIL.Image.open(image_path).convert('RGB')
    face = np.array(face)

    nx = 2
    ny = 2
    img_size_y = face.shape[0] // ny
    img_size_x = face.shape[1] // nx
    predictions = []
    for i_y in range(ny):
        for i_x in range(nx):
            y1 = i_y * img_size_y
            y2 = (i_y + 1) * img_size_y
            x1 = i_x * img_size_x
            x2 = (i_x + 1) * img_size_x
            if i_y == ny - 1 and i_x == nx - 1:
                img = face[y1:, x1:, :]
            if i_y != ny - 1 and i_x == nx - 1:
                img = face[y1:y2, x1:, :]
            if i_y == ny - 1 and i_x != nx - 1:
                img = face[y1:, x1:x2, :]
            if i_y != ny - 1 and i_x != nx - 1:
                img = face[y1:y2, x1:x2, :]

            img = PIL.Image.fromarray(img).convert('RGB')
            img = pil2tensor(img, np.float32)
            img = img.div_(255)
            img = Image(img)

            pred_class, pred_idx, outputs = learn.predict(img)
            output_prediction = outputs.detach().numpy()
            predictions.append(output_prediction[0])

    y1, y2 = (face.shape[0] // 2) - (face.shape[0] // 4), (face.shape[0] // 2) + (face.shape[0] // 4)
    x1, x2 = (face.shape[1] // 2) - (face.shape[1] // 4), (face.shape[1] // 2) + (face.shape[1] // 4)
    img = face[y1:y2, x1:x2, :]
    img = PIL.Image.fromarray(img).convert('RGB')
    img = pil2tensor(img, np.float32)
    img = img.div_(255)
    img = Image(img)

    pred_class, pred_idx, outputs = learn.predict(img)
    output_prediction = outputs.detach().numpy()
    predictions.append(output_prediction[0])

    predictions = np.asarray(predictions)
    return predictions

if __name__ == '__main__':

    test_path = '/mnt/RAID5/users/alfonsomedela/projects/kaggle_comp/Projects/mask_detection/data/test/'
    path = '/mnt/RAID5/users/alfonsomedela/projects/kaggle_comp/Projects/mask_detection/data/'
    sub = pd.read_csv(path + 'sample_sub_v2.csv')

    output = '/mnt/RAID5/users/alfonsomedela/projects/kaggle_comp/Projects/mask_detection/perfect_model/confident_prediction/'


    for i in range(len(sub)):
        filename = sub['image'][i]
        filename = filename[5:-5].split('.')[0]
        image_path = glob.glob(test_path + '*' + filename + '*')[0]

        # predict 5 tiles
        predictions = get_predictions(image_path)

        # tile predictions stats
        min_pred = np.min(predictions)
        max_pred = np.max(predictions)
        mean_pred = np.mean(predictions)

        # predict whole image
        img = open_image(image_path)
        pred_class, pred_idx, outputs = learn.predict(img)
        output_prediction = outputs.detach().numpy()

        # prepare input to rf
        input_x = [[np.min(predictions), np.mean(predictions), max_pred, output_prediction[0]]]
        res = confidence_model.predict(input_x)[0]

        sub['target'][i] = str(res)

    sub.to_csv('confident_submission.csv', index=False)





