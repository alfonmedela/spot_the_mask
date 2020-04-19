import matplotlib
matplotlib.use('Agg')
from fastai.vision import *
import glob
import dlib
from imutils import face_utils
import cv2

torch.cuda.set_device(0)

# Here comes the BEST exported model
learner_path = 'ROOT_PATH/weights/'
learn = load_learner(learner_path)

def get_face(path):
    input_img = cv2.imread(path)

    img_h, img_w, _ = np.shape(input_img)
    ad = 0.3

    gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    face_detect = dlib.get_frontal_face_detector()
    rects = face_detect(gray_img, 1)

    faces = []
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

        xw1 = max(int(x1 - ad * w), 0)
        yw1 = max(int(y1 - ad * h), 0)
        xw2 = min(int(x2 + ad * w), img_w - 1)
        yw2 = min(int(y2 + ad * h), img_h - 1)

        faces.append(input_img[yw1:yw2, xw1:xw2, :])
    return faces

def get_predictions(image_path):

    '''
    :param image_path: input path of the image here
    :return: predictions for the 5 tiles
    '''

    main_image = PIL.Image.open(image_path).convert('RGB')
    main_image = np.array(main_image)

    nx = 2
    ny = 2
    img_size_y = main_image.shape[0] // ny
    img_size_x = main_image.shape[1] // nx
    predictions = []
    for i_y in range(ny):
        for i_x in range(nx):
            y1 = i_y * img_size_y
            y2 = (i_y + 1) * img_size_y
            x1 = i_x * img_size_x
            x2 = (i_x + 1) * img_size_x
            if i_y == ny - 1 and i_x == nx - 1:
                img = main_image[y1:, x1:, :]
            if i_y != ny - 1 and i_x == nx - 1:
                img = main_image[y1:y2, x1:, :]
            if i_y == ny - 1 and i_x != nx - 1:
                img = main_image[y1:, x1:x2, :]
            if i_y != ny - 1 and i_x != nx - 1:
                img = main_image[y1:y2, x1:x2, :]

            img = PIL.Image.fromarray(img).convert('RGB')
            img = pil2tensor(img, np.float32)
            img = img.div_(255)
            img = Image(img)

            pred_class, pred_idx, outputs = learn.predict(img)
            output_prediction = outputs.detach().numpy()
            predictions.append(output_prediction[0])

    # CENTRAL CROP
    y1, y2 = (main_image.shape[0] // 2) - (main_image.shape[0] // 4), (main_image.shape[0] // 2) + (main_image.shape[0] // 4)
    x1, x2 = (main_image.shape[1] // 2) - (main_image.shape[1] // 4), (main_image.shape[1] // 2) + (main_image.shape[1] // 4)
    img = main_image[y1:y2, x1:x2, :]
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

    # path to train data
    path = 'ROOT_PATH/train/'

    # subfolders in train data
    folders = ['mask/', 'no_mask/']

    data = []
    n_class = 0
    for folder in folders:

        images = glob.glob(path + folder + '*')
        for image_path in images:

            faces = get_face(image_path)
            if len(faces) == 0:

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
                output_prediction = output_prediction[0]

                data.append([min_pred, mean_pred, max_pred, output_prediction, n_class])
        n_class += 1

    data = np.asarray(data)
    np.save('training_data', data)










