# -*- coding: utf-8 -*-
"""
@Author  :   Sayed Nadim
@License :   (C) Copyright 2024
@Contact :   sayednadim@deltax.ai
@Software:   VSCode
@File    :   python_version.py
@Time    :   2024-08-22 17:49
@Desc    :   The main purpose of this code is to develop an ML framework for neural network inference. 
             Assumptions:
                1. The model is already trained, optimized, and probably quantized.
                2. The model weights are in an inter-exchangable format such as ONNX. I have used the ONNX loading logic, which can replaced by TFLite loading logic.
                3. I have used an image-based classification model as an example. It can be extended to other downstream regression tasks such as segmentation, depth estimation etc.
                4. The framework assumes cpu-based inference, which can be extended to GPU-based inference.
            Limitations:
                1. This is not a full codebase to run.
                2. Some logic are simplified version of a potential bigger logic flow.
"""


import onnxruntime as ort
import numpy as np
import cv2 


class Preprocessor:
    """ Preprocessing class for input data preparation. 
        I assume that, the larger framework may pass the image path as an argument to the inference
        pipeline. If the input is an image or tensor or even a buffer, we can easily modify it. 
    """
    def preprocess(self, image_path, input_shape):
        """
        Args:
            image_path (str): path of the image. Expected is str. However, it can be an image. We can do a type check.
            input_shape (list of ints): shape of the image. ONNX is good when the size is fixed.

        Returns:
            image (numpy array BCHW): preprocessed image.
        """
        image = cv2.imread(image_path) # used CV2 for simplicity. Can be extended to gstreamer (https://github.com/jackersson/gstreamer-python) or V4l2 (didn't try yet).
        # simple path check, can be extended to full try-except logic
        if image is None:
            print("Failed to load image.")
            return None

        image = cv2.resize(image, (input_shape[2], input_shape[1]))
        image = image.astype(np.float32) / 255.0 # normalization. My assumption is an 8-bit RGB image. If not, we may change it.
        image = np.transpose(image, (2, 0, 1)) # if RGB image. If gray, then we need to extend the dims.
        image = np.expand_dims(image, axis=0) # introducing batch dim. 
        return image

class Postprocessor:
    """ Postprocessing class.
        It will find the argmax of the output. 
        Assumption: output the max probable class. Can be extended to top-k class (https://github.com/numpy/numpy/issues/15128).
    """
    def postprocess(self, output):
        return np.argmax(output)


class Model:
    def __init__(self, model_path):
        """ Simple model loading class.
        I assumed a pre-trained ONNX model to be loaded.
        Args:
            model_path (str): path of the ONNX model. 
        """
        self.session = ort.InferenceSession(model_path) # ort session for inference.

    def run_inference(self, input_tensor):
        inputs = {self.session.get_inputs()[0].name: input_tensor} 
        outputs = self.session.run(None, inputs)
        return outputs[0]


############################# Example run ##########################

def main():
    model_path = "model.onnx"
    image_path = "example.jpg"
    input_shape = (1, 3, 224, 224)

    model = Model(model_path)
    preprocessor = Preprocessor()
    postprocessor = Postprocessor()

    input_tensor = preprocessor.preprocess(image_path, input_shape)
    output_tensor = model.run_inference(input_tensor)
    predicted_class = postprocessor.postprocess(output_tensor)

    print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    main()
