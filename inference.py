import cv2
import numpy as np
import pandas as pd

def prediction(img_array, model, model_seg):
    mask, image_id, has_mask = [], [], []
    
    img = img_array * 1./255.
    img = cv2.resize(img, (256,256))
    img = np.array(img, dtype=np.float64)
    img = np.reshape(img, (1,256,256,3))
    
    is_defect = model.predict(img)
    
    if np.argmax(is_defect) == 0:
        has_mask.append(0)
        mask.append(np.zeros((1,256,256,1)))
    else:
        X = np.empty((1, 256, 256, 3))
        img -= img.mean()
        img /= img.std()
        X[0, ] = img
        
        predict = model_seg.predict(X)
        
        if predict.round().astype(int).sum() == 0:
            has_mask.append(0)
            mask.append(np.zeros((1,256,256,1)))
        else:
            has_mask.append(1)
            mask.append(predict)
    
    return pd.DataFrame({'predicted_mask': mask, 'has_mask': has_mask})
