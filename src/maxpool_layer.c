#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    int start = (l.size - 1) / 2;
    for(int img_i = 0; img_i < in.rows; img_i++) {
        float* inImage = &in.data[img_i * in.cols];
        float* outImage = &out.data[img_i * out.cols];
        for(int c = 0; c < l.channels; c++) {
            // printf("Channel: %d --------------------------- \n", c);
            float* inChanData = &inImage[c * l.width * l.height];
            float* outChanData = &outImage[c * outw * outh];
            for(int rowSteps = 0; rowSteps < l.height; rowSteps += l.stride) {
                for(int colSteps = 0; colSteps < l.width; colSteps += l.stride) {
                    float max = -FLT_MAX;
                    for(int i = 0; i < l.size; i++) {
                        for(int j = 0; j < l.size; j++) {
                            int offset_i = i - start + rowSteps;
                            int offset_j = j - start + colSteps;

                            int image_index = offset_i * l.width + offset_j;
                            if (offset_i < 0 || offset_j < 0 || offset_i >= l.height || offset_j >= l.width){
                                continue;
                            } else {
                                // printf("i %d, j %d, index: %d, val: %f\n", offset_i, offset_j, image_index, inChanData[image_index]);
                                if (inChanData[image_index] > max) {
                                    max = inChanData[image_index];
                                }
                            }
                        }
                    }
                    int outIdx = (rowSteps / l.stride) * outw + (colSteps / l.stride);
                    // printf("Setting at index: %d max: %f\n", outIdx, max);
                    outChanData[outIdx] = max;
                }
            }
        }
    }

    l.in[0] = in;
    free_matrix(l.out[0]);
    l.out[0] = out;
    free_matrix(l.delta[0]);
    l.delta[0] = make_matrix(out.rows, out.cols);
    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix prev_delta: error term for the previous layer
void backward_maxpool_layer(layer l, matrix prev_delta)
{
    matrix in    = l.in[0];
    // matrix out   = l.out[0];
    matrix delta = l.delta[0];

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;

    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.

    int start = (l.size - 1) / 2;
    for(int img_i = 0; img_i < in.rows; img_i++) {
        float* inImage = &in.data[img_i * in.cols];
        float* prevDeltaImage = &prev_delta.data[img_i * prev_delta.cols];
        float* deltaImage = &delta.data[img_i * delta.cols];
        for(int c = 0; c < l.channels; c++) {
            // printf("Channel: %d --------------------------- \n", c);
            float* inChanData = &inImage[c * l.width * l.height];
            float* prevDeltaChanData = &prevDeltaImage[c * l.width * l.height];
            float* deltaChanData = &deltaImage[c * outw * outh];
            for(int rowSteps = 0; rowSteps < l.height; rowSteps += l.stride) {
                for(int colSteps = 0; colSteps < l.width; colSteps += l.stride) {
                    float max = -FLT_MAX;
                    int max_idx_i = -1;
                    int max_idx_j = -1;
                    for(int i = 0; i < l.size; i++) {
                        for(int j = 0; j < l.size; j++) {
                            int offset_i = i - start + rowSteps;
                            int offset_j = j - start + colSteps;

                            int image_index = offset_i * l.width + offset_j;
                            if (offset_i < 0 || offset_j < 0 || offset_i >= l.height || offset_j >= l.width){
                                continue;
                            } else {
                                // printf("i %d, j %d, index: %d, val: %f\n", offset_i, offset_j, image_index, inChanData[image_index]);
                                if (inChanData[image_index] > max) {
                                    max = inChanData[image_index];
                                    max_idx_i = offset_i;
                                    max_idx_j = offset_j;
                                }
                            }
                        }
                    }
                    int outIdx = (rowSteps / l.stride) * outw + (colSteps / l.stride);
                    // printf("Setting at index: %d max: %f\n", outIdx, max);
                    prevDeltaChanData[max_idx_i * l.width + max_idx_j] += deltaChanData[outIdx];
                }
            }
        }
    }
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay)
{
}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.in = calloc(1, sizeof(matrix));
    l.out = calloc(1, sizeof(matrix));
    l.delta = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}
