#include "uwnet.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>

float eps = 0.0001;

matrix mean(matrix x, int spatial)
{
    matrix m = make_matrix(1, x.cols/spatial);
    int i, j;
    for(i = 0; i < x.rows; ++i){
        for(j = 0; j < x.cols; ++j){
            m.data[j/spatial] += x.data[i*x.cols + j];
        }
    }
    for(i = 0; i < m.cols; ++i){
        m.data[i] = m.data[i] / x.rows / spatial;
    }
    return m;
}

matrix variance(matrix x, matrix m, int spatial)
{
    matrix v = make_matrix(1, x.cols/spatial);

    int i, j;
    for(i = 0; i < x.rows; i++) {
        for(j = 0; j < x.cols; j++) {
            v.data[j / spatial] += powf(x.data[i * x.cols + j] - m.data[j / spatial], 2);
        }
    }

    for(i = 0; i < v.cols; i++) {
        v.data[i] = v.data[i] / x.rows / spatial;
    }

    return v;
}

matrix normalize(matrix x, matrix m, matrix v, int spatial)
{
    matrix norm = make_matrix(x.rows, x.cols);
    // TODO: 7.2 - normalize array, norm = (x - mean) / sqrt(variance + eps)
    int i, j;
    for(i = 0; i < x.rows; i++) {
        for(j = 0; j < x.cols; j++) {
            float mean = m.data[j / spatial];
            float variance = v.data[j / spatial];
            float data = x.data[i * x.cols + j];
            norm.data[i * x.cols + j] = (data - mean) / sqrtf(variance + eps);
        }
    }
    return norm;
}

matrix batch_normalize_forward(layer l, matrix x)
{
    float s = .1;
    int spatial = x.cols / l.rolling_mean.cols;
    if (x.rows == 1){
        return normalize(x, l.rolling_mean, l.rolling_variance, spatial);
    }
    matrix m = mean(x, spatial);
    matrix v = variance(x, m, spatial);

    matrix x_norm = normalize(x, m, v, spatial);

    scal_matrix(1-s, l.rolling_mean);
    axpy_matrix(s, m, l.rolling_mean);

    scal_matrix(1-s, l.rolling_variance);
    axpy_matrix(s, v, l.rolling_variance);

    free_matrix(m);
    free_matrix(v);

    free_matrix(l.x[0]);
    l.x[0] = x;

    return x_norm;
}


matrix delta_mean(matrix d, matrix variance, int spatial)
{ 
    matrix dm = make_matrix(1, variance.cols);
    // TODO: 7.3 - calculate dL/dmean
    int i, j;
    for(i = 0; i < d.rows; i++) {
        for(j = 0; j < d.cols; j++) {
            float var = variance.data[j / spatial];
            dm.data[j / spatial] += d.data[i * d.cols + j] * (-1.0/sqrtf(var + eps));
        }
    }

    // for(i = 0; i < dm.cols; i++) {
    //     dm.data[i] = -1.0*dm.data[i]/sqrtf(variance.data[i] + eps);
    // }
    return dm;
}

matrix delta_variance(matrix d, matrix x, matrix mean, matrix variance, int spatial)
{
    matrix dv = make_matrix(1, variance.cols);
    // TODO: 7.4 - calculate dL/dvariance
    int i, j;
    for(i = 0; i < d.rows; i++) {
        for(j = 0; j < d.cols; j++) {
            dv.data[j / spatial] += d.data[i * d.cols + j] 
                                    * (x.data[i * d.cols + j] - mean.data[j / spatial])
                                    * -0.5
                                    * powf(variance.data[j / spatial] + eps, -1.5);
        }
    }
    return dv;
}

matrix delta_batch_norm(matrix d, matrix dm, matrix dv, matrix mean, matrix variance, matrix x, int spatial)
{
    int i, j;
    matrix dx = make_matrix(d.rows, d.cols);
    // TODO: 7.5 - calculate dL/dx
    for(i = 0; i < d.rows; i++) {
        for(j = 0; j < d.cols; j++) {
            dx.data[i * d.cols + j] = d.data[i * d.cols + j] * (1.0 / (sqrtf(variance.data[j / spatial] + eps))) +
                                        dv.data[j / spatial] * (2.0 * (x.data[i * d.cols + j] - mean.data[j / spatial]) / x.rows / spatial) +
                                        dm.data[j / spatial] * (1.0 / x.rows / spatial);
        }
    }
    return dx;
}

matrix batch_normalize_backward(layer l, matrix d)
{
    // printf("%s cols: %d mean_cols %d\n", "here1", d.cols, l.rolling_mean.cols);
    int spatial = d.cols / l.rolling_mean.cols;
    // printf("%s\n", "here2");
    matrix x = l.x[0];

    // printf("%s asdf %d\n ", "here3", spatial);
    matrix m = mean(x, spatial);
    // printf("%s\n", "here4");
    matrix v = variance(x, m, spatial);

    // printf("%s\n", "mean");
    matrix dm = delta_mean(d, v, spatial);
    // printf("%s\n", "variance");
    matrix dv = delta_variance(d, x, m, v, spatial);
    // printf("%s\n", "norm");
    matrix dx = delta_batch_norm(d, dm, dv, m, v, x, spatial);
    // printf("%s\n", "done");

    free_matrix(m);
    free_matrix(v);
    free_matrix(dm);
    free_matrix(dv);

    return dx;
}