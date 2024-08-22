/*
@Author  :   Sayed Nadim
@License :   (C) Copyright 2024
@Contact :   sayednadim@deltax.ai
@Software:   Visual Studio
@File    :   cpp_with_gstreamer.cpp
@Time    :   2024-08-22 11:49
@Desc    :   The main purpose of this code is to develop a C++ ML framework for neural network inference.
             Why gstreamer version:
                1. Many boards don't support opencv for image streaming. I had a rough experience with TI board. 
                2. Gstreamer is ok for low-level handing.
                3. Not resource intensive.
             Assumptions:
                1. The model is already trained, optimized, and possibly quantized.
                2. The model weights are in an interchangeable format such as ONNX. I have used the ONNX Runtime API, but this can be replaced by TFLite or other frameworks.
                3. The example demonstrates an image-based classification model. This framework can be extended to support other tasks like segmentation, depth estimation, etc.
                4. The framework assumes CPU-based inference but can be extended to support GPU-based inference.
            Limitations:
                1. This is not a full codebase for production use.
                2. Some logic is simplified for clarity and could be expanded in a larger framework.
                3. I am not an expert in Gstreamer. I used some codes from my previous codebase.
*/

#include <onnxruntime/core/session/onnxruntime_cxx_api.h> // ONNX Runtime C++ API for neural network inference
#include <gst/gst.h>                                       // GStreamer for multimedia handling and processing
#include <vector>                                          // For dynamic arrays
#include <string>                                          // For handling text data
#include <iostream>                                        // For input/output operations
#include <algorithm>                                       // For standard algorithms like max_element
#include <cstring>                                         // For memory manipulation


// If json is preferred for the config-type initialization, we can use json (https://github.com/nlohmann/json)

// Model Class: Manages the loading and inference of the ONNX model
class Model {
private:
    Ort::Env env;                 // ONNX Runtime environment for managing inference session
    Ort::Session session;         // ONNX Runtime session for running the model
    Ort::SessionOptions session_options; // Configuration options for the inference session

public:
    /*
    Constructor for loading a pre-trained ONNX model.
    Assumption: The model is already trained and saved in ONNX format.

    Args:
        model_path (const std::string&): Path to the ONNX model file.
    */
    Model(const std::string& model_path)
        : env(ORT_LOGGING_LEVEL_WARNING, "Model"), session_options(), session(nullptr) {
        session_options.SetIntraOpNumThreads(1); // Set the number of threads for inference. Adjust based on hardware.
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
        Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(input_tensor.data()), input_tensor.size(), input_shape.data(), input_shape.size());

        std::vector<int64_t> output_shape = {1, 1000}; // Example output shape, should be adapted to your model.
        Ort::Value output_tensor_ort = Ort::Value::CreateTensor<float>(memory_info, output_shape.data(), output_shape.size()); // Create output tensor.

        const char* input_names[] = {"input"};  // Input node name in the ONNX model.
        const char* output_names[] = {"output"}; // Output node name in the ONNX model.

        session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_ort, 1, output_names, &output_tensor_ort, 1); // Run the inference session.

        float* floatarr = output_tensor_ort.GetTensorMutableData<float>(); // Get a pointer to the output data.
        return std::vector<float>(floatarr, floatarr + output_shape[1]); // Convert output data to a vector of floats.
    }
};

// Preprocessor Class: Handles input data preparation using GStreamer
class Preprocessor {
public:
    /*
    Preprocessing function for input data preparation using GStreamer.
    Assumption: The function processes an image from the file path to produce a tensor in BCHW format.
    
    Args:
        image_path (const std::string&): Path to the image file. Expected as a string.
        input_shape (const std::vector<int>&): Shape of the input image. ONNX models typically expect fixed input sizes.
    
    Returns:
        std::vector<float>: Preprocessed image as a vector of floats.
    */
    std::vector<float> preprocess(const std::string& image_path, const std::vector<int>& input_shape) {
        GstElement *pipeline, *source, *capsfilter, *convert, *sink;
        GstBus *bus;
        GstMessage *msg;
        GstCaps *caps;
        GstSample *sample = nullptr;
        GstBuffer *buffer;
        GstMapInfo map;

        gst_init(nullptr, nullptr); // Initialize GStreamer

        // Create GStreamer elements: file source, caps filter, video converter, and sink.
        pipeline = gst_pipeline_new("image-pipeline");
        source = gst_element_factory_make("filesrc", "source");
        capsfilter = gst_element_factory_make("capsfilter", "capsfilter");
        convert = gst_element_factory_make("videoconvert", "convert");
        sink = gst_element_factory_make("appsink", "sink");

        // Check if elements were created successfully.
        if (!pipeline || !source || !capsfilter || !convert || !sink) {
            std::cerr << "Not all elements could be created." << std::endl;
            return {}; // Return an empty vector if element creation fails.
        }

        // Set the properties for the GStreamer pipeline elements.
        g_object_set(source, "location", image_path.c_str(), nullptr); // Set the image file path.
        caps = gst_caps_new_simple("video/x-raw", // Specify the desired image properties.
                                   "format", G_TYPE_STRING, "RGB",
                                   "width", G_TYPE_INT, input_shape[2],
                                   "height", G_TYPE_INT, input_shape[1],
                                   "framerate", GST_TYPE_FRACTION, 1, 1,
                                   nullptr);
        g_object_set(capsfilter, "caps", caps, nullptr); // Apply caps filter to ensure image format.
        gst_caps_unref(caps); // Unreference the caps object after setting it.

        // Build the pipeline by linking elements.
        gst_bin_add_many(GST_BIN(pipeline), source, capsfilter, convert, sink, nullptr);
        gst_element_link_many(source, capsfilter, convert, sink, nullptr);

        // Set up the sink to emit signals and disable synchronization.
        g_object_set(sink, "emit-signals", TRUE, "sync", FALSE, nullptr);
        g_signal_connect(sink, "new-sample", G_CALLBACK(on_new_sample), &sample); // Connect signal for new samples.

        gst_element_set_state(pipeline, GST_STATE_PLAYING); // Start the pipeline.
        bus = gst_element_get_bus(pipeline); // Get the message bus to monitor pipeline events.
        msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE, GST_MESSAGE_ERROR | GST_MESSAGE_EOS); // Wait for error or EOS.

        // Handle any errors that occurred during pipeline execution.
        if (msg != nullptr) {
            if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                GError *err;
                gchar *debug_info;
                gst_message_parse_error(msg, &err, &debug_info);
                std::cerr << "Error received from element " << GST_OBJECT_NAME(msg->src) << ": " << err->message << std::endl;
                g_clear_error(&err);
                g_free(debug_info);
            }
            gst_message_unref(msg); // Free the message object.
        }

        std::vector<float> input_tensor;
        // Process the sample and map it to the input tensor if available.
        if (sample != nullptr) {
            buffer = gst_sample_get_buffer(sample); // Get buffer from sample.
            if (gst_buffer_map(buffer, &map, GST_MAP_READ)) { // Map buffer for reading.
                int width = input_shape[2];
                int height = input_shape[1];
                int channels = input_shape[0];

                input_tensor.resize(width * height * channels); // Resize tensor to fit the image.

                uint8_t* data = map.data; // Get data pointer from the mapped buffer.

                // Rearrange and normalize image data into BCHW format.
                for (int c = 0; c < channels; ++c) {
                    for (int y = 0; y < height; ++y) {
                        for (int x = 0; x < width; ++x) {
                            int idx = (y * width + x) * channels + c; // Index for tensor.
                            int offset = y * width * channels + x * channels + c; // Offset in image data.
                            input_tensor[idx] = data[offset] / 255.0f; // Normalize the pixel value.
                        }
                    }
                }

                gst_buffer_unmap(buffer, &map); // Unmap the buffer after reading.
            }
        }

        gst_element_set_state(pipeline, GST_STATE_NULL); // Set pipeline state to NULL after processing.
        gst_object_unref(pipeline); // Unreference the pipeline to clean up.

        return input_tensor; // Return the preprocessed tensor.
    }

private:
    /*
    Callback function to handle new samples in the GStreamer pipeline.
    Assumption: The sample contains the processed image data from the pipeline.
    
    Args:
        sink (GstElement*): The sink element from the GStreamer pipeline.
        user_data (gpointer): Pointer to user data, used to store the sample.
    
    Returns:
        GstFlowReturn: Return status for GStreamer, indicating success or failure.
    */
    static GstFlowReturn on_new_sample(GstElement* sink, gpointer user_data) {
        GstSample** sample = reinterpret_cast<GstSample**>(user_data); // Cast user data to sample pointer.
        *sample = gst_app_sink_pull_sample(GST_APP_SINK(sink)); // Pull the sample from the sink.
        return GST_FLOW_OK; // Indicate that the sample was successfully handled.
    }
};

// Postprocessor Class: Handles post-inference operations such as output tensor interpretation
class Postprocessor {
public:
    /*
    Postprocessing function to interpret the model output.
    Assumption: The model output is a probability distribution over classes.
    
    Args:
        output (const std::vector<float>&): Model output tensor as a vector of floats.
    
    Returns:
        int: Predicted class index. Didn't try the top-k.
    */
    int postprocess(const std::vector<float>& output) {
        return std::max_element(output.begin(), output.end()) - output.begin(); // Return the index of the max value.
    }
};

/*
Exmaple run
Assumption: The input image and model paths are valid and the input shape matches the model requirements.
*/
int main() {
    std::string model_path = "model.onnx";  // Path to the ONNX model file.
    std::string image_path = "example.jpg"; // Path to the input image file.
    std::vector<int> input_shape = {3, 224, 224}; // Expected input shape in BCHW format.

    Model model(model_path); // Load the model.
    Preprocessor preprocessor; // Create a preprocessor instance.
    Postprocessor postprocessor; // Create a postprocessor instance.

    std::vector<float> input_tensor = preprocessor.preprocess(image_path, input_shape); // Preprocess the input image.
    std::vector<float> output_tensor = model.run_inference(input_tensor, {1, 3, 224, 224}); // Run inference.
    int predicted_class = postprocessor.postprocess(output_tensor); // Postprocess the output to get the predicted class.

    std::cout << "Predicted class: " << predicted_class << std::endl; // Output the predicted class.

    return 0; // Indicate successful execution.
}
