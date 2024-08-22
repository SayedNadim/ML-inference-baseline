/*
@Author  :   Sayed Nadim
@License :   (C) Copyright 2024
@Contact :   sayednadim@deltax.ai
@Software:   Visual Studio
@File    :   cpp_with_opencv.cpp
@Time    :   2024-08-21 18:49
@Desc    :   The main purpose of this code is to develop a C++ ML framework for neural network inference.
             Why opencv version:
                1. During the meeting, I learned that some functions of opencv were used in the framework.
                2. My assumption is using opencv for image manipulation is not an issue at this moment.
             Assumptions:
                1. The model is already trained, optimized, and possibly quantized.
                2. The model weights are in an interchangeable format such as ONNX. I have used the ONNX Runtime API, but this can be replaced by TFLite or other frameworks.
                3. The example demonstrates an image-based classification model. This framework can be extended to support other tasks like segmentation, depth estimation, etc.
                4. The framework assumes CPU-based inference but can be extended to support GPU-based inference.
            Limitations:
                1. This is not a full codebase for production use.
                2. Some logic is simplified for clarity and could be expanded in a larger framework.
*/

#include <onnxruntime/core/session/onnxruntime_cxx_api.h> // ONNX Runtime C++ API for neural network inference. It will show error as I didn't build it.
#include <opencv2/opencv.hpp>                            // OpenCV library for image processing. Same with onnx. 
#include <vector>                                        // For dynamic array support
#include <string>                                        // For handling text data
#include <iostream>                                      // For input/output operations

// If json is preferred for the config-type initialization, we can use json (https://github.com/nlohmann/json)

// Preprocessor Class: Prepares input data for the model
class Preprocessor {
public:
    /*
    Preprocessing function for input data preparation.
    Assumption: The larger framework may pass the image path as an argument. 
    The function could be modified to handle images, tensors, or buffers as inputs.
    
    Args:
        image_path (const std::string&): Path to the image file. Expected as a string.
        input_shape (const std::vector<int>&): Shape of the input image. ONNX models typically expect fixed input sizes.
    
    Returns:
        std::vector<float>: Preprocessed image in BCHW format.
    */
    std::vector<float> preprocess(const std::string& image_path, const std::vector<int>& input_shape) {
        cv::Mat image = cv::imread(image_path); // Using OpenCV for simplicity. Can be replaced with GStreamer or other libraries.

        if (image.empty()) { // Simple error check; can be extended to more robust error handling.
            std::cerr << "Failed to load image." << std::endl;
            return {}; // Return an empty vector if the image cannot be loaded.
        }

        cv::resize(image, image, cv::Size(input_shape[2], input_shape[1])); // Resize to match the model's expected input shape.
        image.convertTo(image, CV_32F, 1.0 / 255.0); // Normalize the image. Assumes an 8-bit RGB image. Adjustments may be needed for other formats.
        
        // Convert to CHW format and introduce a batch dimension (1 x 3 x H x W).
        std::vector<float> input_tensor(input_shape[0] * input_shape[1] * input_shape[2]);
        std::memcpy(input_tensor.data(), image.data, input_tensor.size() * sizeof(float)); 

        return input_tensor; // Return the preprocessed image tensor.
    }
};

// Postprocessor Class: Interprets the model's output
class Postprocessor {
public:
    /*
    Postprocessing function to extract the predicted class from model output.
    Assumption: The function returns the class with the highest probability. 
    This could be extended to return top-k classes or confidence scores. Didn't try.
    
    Args:
        output (const std::vector<float>&): Model output as a vector of floats.
    
    Returns:
        int: Index of the predicted class.
    */
    int postprocess(const std::vector<float>& output) {
        return std::max_element(output.begin(), output.end()) - output.begin(); // Return the index of the maximum element.
    }
};

// Model Class: Manages model loading and inference
class Model {
public:
    /*
    Constructor for loading a pre-trained ONNX model.
    Assumption: The model is already trained and saved in the ONNX format.

    Args:
        model_path (const std::string&): Path to the ONNX model file.
    */
    Model(const std::string& model_path)
        : env(ORT_LOGGING_LEVEL_WARNING, "Model"), // Initialize ONNX Runtime environment with warning level logging.
          session_options(),                       // Initialize session options with default settings.
          session(nullptr) {                       // Initialize session to null before loading the model.

        session_options.SetIntraOpNumThreads(1); // Set number of threads for inference. Adjust as needed.
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED); // Enable extended graph optimizations.
        session = Ort::Session(env, model_path.c_str(), session_options); // Load the ONNX model into the session.
    }

    /*
    Function to run inference on the preprocessed input tensor.

    Args:
        input_tensor (const std::vector<float>&): Preprocessed input data as a vector of floats.
        input_shape (const std::vector<int64_t>&): Shape of the input tensor.

    Returns:
        std::vector<float>: Output tensor from the model.
    */
    std::vector<float> run_inference(const std::vector<float>& input_tensor, const std::vector<int64_t>& input_shape) {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault); // Define memory info for CPU allocation.

        // Create ONNX input tensor from the input data.
        Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(memory_info, 
                                                                      const_cast<float*>(input_tensor.data()), 
                                                                      input_tensor.size(), 
                                                                      input_shape.data(), 
                                                                      input_shape.size());

        std::vector<int64_t> output_shape = {1, 1000}; // Example output shape, adjust based on model architecture.
        Ort::Value output_tensor_ort = Ort::Value::CreateTensor<float>(memory_info, 
                                                                       output_shape.data(), 
                                                                       output_shape.size()); // Create output tensor.

        const char* input_names[] = {"input"}; // Input node name in the ONNX model.
        const char* output_names[] = {"output"}; // Output node name in the ONNX model.

        session.Run(Ort::RunOptions{nullptr}, 
                    input_names, &input_tensor_ort, 1, 
                    output_names, &output_tensor_ort, 1); // Run the inference session.

        float* floatarr = output_tensor_ort.GetTensorMutableData<float>(); // Get pointer to output data.
        return std::vector<float>(floatarr, floatarr + output_shape[1]); // Convert output data to a vector of floats.
    }

private:
    Ort::Env env;                         // ONNX Runtime environment for managing inference session.
    Ort::Session session;                 // ONNX Runtime session for running the model.
    Ort::SessionOptions session_options;  // Configuration options for the inference session.
};

/* Example Run */

int main() {
    std::string model_path = "model.onnx"; // Path to the ONNX model file.
    std::string image_path = "example.jpg"; // Path to the input image file.
    std::vector<int> input_shape = {3, 224, 224}; // Expected input shape for the model (channels, height, width).

    Model model(model_path); // Instantiate the model object.
    Preprocessor preprocessor; // Instantiate the preprocessor object.
    Postprocessor postprocessor; // Instantiate the postprocessor object.

    std::vector<float> input_tensor = preprocessor.preprocess(image_path, input_shape); // Preprocess the input image.
    std::vector<float> output_tensor = model.run_inference(input_tensor, {1, 3, 224, 224}); // Run inference and get output tensor.
    int predicted_class = postprocessor.postprocess(output_tensor); // Postprocess the output to get the predicted class.

    std::cout << "Predicted class: " << predicted_class << std::endl; // Output the predicted class.

    return 0; // Indicate successful execution.
}
