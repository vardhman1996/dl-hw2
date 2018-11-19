#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include "uwnet.h"
#include "matrix.h"
#include "image.h"
#include "test.h"
#include "args.h"

int tests_total = 0;
int tests_fail = 0;


int within_eps(float a, float b){
    return a-EPS<b && b<a+EPS;
}

int same_matrix(matrix a, matrix b)
{
    int i;
    if(a.rows != b.rows || a.cols != b.cols) return 0;
    for(i = 0; i < a.rows*a.cols; ++i){
        if(!within_eps(a.data[i], b.data[i])) return 0;
    }
    return 1;
}

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void test_copy_matrix()
{
    matrix a = random_matrix(32, 64, 10);
    matrix c = copy_matrix(a);
    TEST(same_matrix(a,c));
    free_matrix(a);
    free_matrix(c);
}

void test_transpose_matrix()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix at = load_matrix("data/test/at.matrix");
    matrix atest = transpose_matrix(a);
    matrix aorig = transpose_matrix(atest);
    TEST(same_matrix(at, atest) && same_matrix(a, aorig));
    free_matrix(a);
    free_matrix(at);
    free_matrix(atest);
    free_matrix(aorig);
}

void test_axpy_matrix()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix y = load_matrix("data/test/y.matrix");
    matrix y1 = load_matrix("data/test/y1.matrix");
    axpy_matrix(2, a, y);
    TEST(same_matrix(y, y1));
    free_matrix(a);
    free_matrix(y);
    free_matrix(y1);
}

void test_matmul()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix b = load_matrix("data/test/b.matrix");
    matrix c = load_matrix("data/test/c.matrix");
    matrix mul = matmul(a, b);
    TEST(same_matrix(c, mul));
    free_matrix(a);
    free_matrix(b);
    free_matrix(c);
    free_matrix(mul);
}

void test_activate_matrix()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix truth_alog = load_matrix("data/test/alog.matrix");
    matrix truth_arelu = load_matrix("data/test/arelu.matrix");
    matrix truth_alrelu = load_matrix("data/test/alrelu.matrix");
    matrix truth_asoft = load_matrix("data/test/asoft.matrix");
    matrix alog = copy_matrix(a);
    // print_matrix(alog);
    activate_matrix(alog, LOGISTIC);
    matrix arelu = copy_matrix(a);
    activate_matrix(arelu, RELU);
    // print_matrix(arelu);
    // print_matrix(truth_arelu);
    matrix alrelu = copy_matrix(a);
    activate_matrix(alrelu, LRELU);
    matrix asoft = copy_matrix(a);
    activate_matrix(asoft, SOFTMAX);
    TEST(same_matrix(truth_alog, alog));
    TEST(same_matrix(truth_arelu, arelu));
    TEST(same_matrix(truth_alrelu, alrelu));
    TEST(same_matrix(truth_asoft, asoft));
    free_matrix(a);
    free_matrix(alog);
    free_matrix(arelu);
    free_matrix(alrelu);
    free_matrix(asoft);
    free_matrix(truth_alog);
    free_matrix(truth_arelu);
    free_matrix(truth_alrelu);
    free_matrix(truth_asoft);
}

void test_gradient_matrix()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix y = load_matrix("data/test/y.matrix");
    matrix truth_glog = load_matrix("data/test/glog.matrix");
    matrix truth_grelu = load_matrix("data/test/grelu.matrix");
    matrix truth_glrelu = load_matrix("data/test/glrelu.matrix");
    matrix truth_gsoft = load_matrix("data/test/gsoft.matrix");
    matrix glog = copy_matrix(a);
    matrix grelu = copy_matrix(a);
    matrix glrelu = copy_matrix(a);
    matrix gsoft = copy_matrix(a);
    gradient_matrix(y, LOGISTIC, glog);
    gradient_matrix(y, RELU, grelu);
    gradient_matrix(y, LRELU, glrelu);
    gradient_matrix(y, SOFTMAX, gsoft);
    TEST(same_matrix(truth_glog, glog));
    TEST(same_matrix(truth_grelu, grelu));
    TEST(same_matrix(truth_glrelu, glrelu));
    TEST(same_matrix(truth_gsoft, gsoft));
    free_matrix(a);
    free_matrix(glog);
    free_matrix(grelu);
    free_matrix(glrelu);
    free_matrix(gsoft);
    free_matrix(truth_glog);
    free_matrix(truth_grelu);
    free_matrix(truth_glrelu);
    free_matrix(truth_gsoft);
}

void test_connected_layer()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix b = load_matrix("data/test/b.matrix");
    matrix dw = load_matrix("data/test/dw.matrix");
    matrix db = load_matrix("data/test/db.matrix");
    matrix delta = load_matrix("data/test/delta.matrix");
    matrix prev_delta = load_matrix("data/test/prev_delta.matrix");
    matrix truth_prev_delta = load_matrix("data/test/truth_prev_delta.matrix");
    matrix truth_dw = load_matrix("data/test/truth_dw.matrix");
    matrix truth_db = load_matrix("data/test/truth_db.matrix");
    matrix updated_dw = load_matrix("data/test/updated_dw.matrix");
    matrix updated_db = load_matrix("data/test/updated_db.matrix");
    matrix updated_w = load_matrix("data/test/updated_w.matrix");
    matrix updated_b = load_matrix("data/test/updated_b.matrix");

    matrix bias = load_matrix("data/test/bias.matrix");
    matrix truth_out = load_matrix("data/test/out.matrix");
    layer l = make_connected_layer(64, 16, LRELU);
    l.w = b;
    l.b = bias;
    l.dw = dw;
    l.db = db;
    matrix out = l.forward(l, a);
    TEST(same_matrix(truth_out, out));


    l.delta[0] = delta;
    l.backward(l, prev_delta);
    TEST(same_matrix(truth_prev_delta, prev_delta));
    TEST(same_matrix(truth_dw, l.dw));
    TEST(same_matrix(truth_db, l.db));

    l.update(l, .01, .9, .01);
    TEST(same_matrix(updated_dw, l.dw));
    TEST(same_matrix(updated_db, l.db));
    TEST(same_matrix(updated_w, l.w));
    TEST(same_matrix(updated_b, l.b));

    free_matrix(a);
    free_matrix(b);
    free_matrix(bias);
    free_matrix(out);
    free_matrix(truth_out);
}

void make_matrix_test()
{
    srand(0);
    matrix a = random_matrix(32, 64, 10);
    matrix b = random_matrix(64, 16, 10);
    matrix at = transpose_matrix(a);
    matrix c = matmul(a, b);
    matrix y = random_matrix(32, 64, 10);
    matrix bias = random_matrix(1, 16, 10);
    matrix dw = random_matrix(64, 16, 10);
    matrix db = random_matrix(1, 16, 10);
    matrix delta = random_matrix(32, 16, 10);
    matrix prev_delta = random_matrix(32, 64, 10);
    matrix y1 = copy_matrix(y);
    axpy_matrix(2, a, y1);
    save_matrix(a, "data/test/a.matrix");
    save_matrix(b, "data/test/b.matrix");
    save_matrix(bias, "data/test/bias.matrix");
    save_matrix(dw, "data/test/dw.matrix");
    save_matrix(db, "data/test/db.matrix");
    save_matrix(at, "data/test/at.matrix");
    save_matrix(delta, "data/test/delta.matrix");
    save_matrix(prev_delta, "data/test/prev_delta.matrix");
    save_matrix(c, "data/test/c.matrix");
    save_matrix(y, "data/test/y.matrix");
    save_matrix(y1, "data/test/y1.matrix");

    matrix alog = copy_matrix(a);
    activate_matrix(alog, LOGISTIC);
    save_matrix(alog, "data/test/alog.matrix");

    matrix arelu = copy_matrix(a);
    activate_matrix(arelu, RELU);
    save_matrix(arelu, "data/test/arelu.matrix");

    matrix alrelu = copy_matrix(a);
    activate_matrix(alrelu, LRELU);
    save_matrix(alrelu, "data/test/alrelu.matrix");

    matrix asoft = copy_matrix(a);
    activate_matrix(asoft, SOFTMAX);
    save_matrix(asoft, "data/test/asoft.matrix");



    matrix glog = copy_matrix(a);
    gradient_matrix(y, LOGISTIC, glog);
    save_matrix(glog, "data/test/glog.matrix");

    matrix grelu = copy_matrix(a);
    gradient_matrix(y, RELU, grelu);
    save_matrix(grelu, "data/test/grelu.matrix");

    matrix glrelu = copy_matrix(a);
    gradient_matrix(y, LRELU, glrelu);
    save_matrix(glrelu, "data/test/glrelu.matrix");

    matrix gsoft = copy_matrix(a);
    gradient_matrix(y, SOFTMAX, gsoft);
    save_matrix(gsoft, "data/test/gsoft.matrix");

    layer l = make_connected_layer(64, 16, LRELU);
    l.w = b;
    l.b = bias;
    l.dw = dw;
    l.db = db;

    matrix out = l.forward(l, a);
    save_matrix(out, "data/test/out.matrix");

    l.delta[0] = delta;
    l.backward(l, prev_delta);
    save_matrix(prev_delta, "data/test/truth_prev_delta.matrix");
    save_matrix(l.dw, "data/test/truth_dw.matrix");
    save_matrix(l.db, "data/test/truth_db.matrix");

    l.update(l, .01, .9, .01);
    save_matrix(l.dw, "data/test/updated_dw.matrix");
    save_matrix(l.db, "data/test/updated_db.matrix");
    save_matrix(l.w, "data/test/updated_w.matrix");
    save_matrix(l.b, "data/test/updated_b.matrix");
}

void test_matrix_speed()
{
    int i;
    int n = 128;
    matrix a = random_matrix(512, 512, 1);
    matrix b = random_matrix(512, 512, 1);
    double start = what_time_is_it_now();
    for(i = 0; i < n; ++i){
        matrix d = matmul(a,b);
        free_matrix(d);
    }
    printf("Matmul elapsed %lf sec\n", what_time_is_it_now() - start);
    start = what_time_is_it_now();
    for(i = 0; i < n; ++i){
        matrix at = transpose_matrix(a);
        free_matrix(at);
    }
    printf("Transpose elapsed %lf sec\n", what_time_is_it_now() - start);
}

void bn_forward() {
    int b = 2;
    float in_data[] = {
        1,2,3,4,5,6,7,8,9,10,11,12
    };
    int dim = 6;
    int spatial = 2;
    matrix x = make_matrix(b, dim);
    x.data = in_data;

    matrix mean_mat = mean(x, spatial);
    matrix variance_mat = variance(x, mean_mat, spatial);
    matrix norm = normalize(x, mean_mat, variance_mat, spatial);

    printf("Printing original data \n");
    print_matrix(x);
    printf("Printing mean matrix with spatial %d \n", spatial);
    print_matrix(mean_mat);
    printf("Printing variance matrix with spatial %d \n", spatial);
    print_matrix(variance_mat);
    printf("Printing normalize matrix with spatial %d \n", spatial);
    print_matrix(norm);


    float d_in[] = {
        1,1,1,1,1,1,1,1,1,1,1,1
    };

    matrix d_mat = make_matrix(b, dim);
    d_mat.data = d_in;
    matrix d_mean = delta_mean(d_mat, variance_mat, spatial);
    matrix d_var = delta_variance(d_mat, x, mean_mat, variance_mat, spatial);
    matrix d_bn = delta_batch_norm(d_mat, d_mean, d_var, mean_mat, variance_mat, x, spatial);


    printf("Printing dmean with spatial %d \n", spatial);
    print_matrix(d_mean);
    printf("Printing dvar with spatial %d \n", spatial);
    print_matrix(d_var);
    printf("Printing dbn with spatial %d \n", spatial);
    print_matrix(d_bn);
}

void test_batch_norm() {
  int channels = 2;
  int batch = 2;
  int width = 7, height = 7;
  int stride = 1, size = 3;
  int cols = width * height * channels;

  // Init.
  matrix in = make_matrix(batch, cols);
  matrix prev_delta = make_matrix(batch, cols);

  // Raw flat data.
  int counter = 0;
  int len = batch * channels * width * height;
  assert(len == 196);
  float data[196] = {
      0,  1,  2,  3,  4,  5,  6,
      7,  8,  9,  10, 11, 12, 13,
      14, 15, 16, 17, 18, 19, 20,
      21, 22, 23, 24, 25, 26, 27,
      28, 29, 30, 31, 32, 33, 34,
      35, 36, 37, 38, 39, 40, 41,
      42, 43, 44, 45, 46, 47, 48,

      0,  1,  2,  3,  4,  5,  6,
      7,  8,  9,  10, 11, 12, 13,
      14, 15, 16, 17, 18, 19, 20,
      21, 22, 23, 24, 25, 26, 27,
      28, 29, 30, 31, 32, 33, 34,
      35, 36, 37, 38, 39, 40, 41,
      42, 43, 44, 45, 46, 47, 48,

      0,  1,  2,  3,  4,  5,  6,
      7,  8,  9,  10, 11, 12, 13,
      14, 15, 16, 17, 18, 19, 20,
      21, 22, 23, 24, 25, 26, 27,
      28, 29, 30, 31, 32, 33, 34,
      35, 36, 37, 38, 39, 40, 41,
      42, 43, 44, 45, 46, 47, 48,

      0,  1,  2,  3,  4,  5,  6,
      7,  8,  9,  10, 11, 12, 13,
      14, 15, 16, 17, 18, 19, 20,
      21, 22, 23, 24, 25, 26, 27,
      28, 29, 30, 31, 32, 33, 34,
      35, 36, 37, 38, 39, 40, 41,
      42, 43, 44, 45, 46, 47, 48,
  };

//    assert(len == 64);
//    float data[64] = {
//        -7,  6, -1,  3,  9,  9,  6, -9,
//         3, -8,  0,  7, 10,  8, -3, 10,
//        -4,  2, -6,  4, -7,  5,  5,  7,
//        -3, -9,  1,  8, -8,  9, -1, -5,
//        -7, 10, -9, -5,  9, -8, -7, 10,
//        -5,  5,  9,  4, 10, -8,  7,  6,
//        -3,  8,  0,  2,  2, -3, -2,  5,
//         4, -6,  7, -3,  1,  4, 10,  0,
//    };

//    assert(len == 2 * 2 * 2 * 2);
//    float data[2 * 2 * 2 * 2] = {
//        1, 2, 3, 4,
//        5, 6, 7, 8,
//
//        1, 2, 3, 4,
//        5, 6, 7, 8,
//    };

  for (int b = 0; b < batch; b++) {
    for (int ch = 0; ch < channels; ch++) {
      for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
          assert(counter < len);

          in.data[b * width * height * channels
                  + ch * (width * height)
                  + r * (width)
                  + c] = data[counter];
          counter += 1;
        }
      }
    }
  }
  print_matrix(in);
  printf("------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
  matrix m = mean(in, width*height);
  matrix var = variance(in, m, width*height);
  matrix norm = normalize(in, m, var, width*height);
  print_matrix(m);
  print_matrix(var);
  print_matrix(norm);

//  matrix d = make_matrix(1, )
//  delta_mean(d, variance, width*height);

  printf("------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
}


void run_tests()
{
    //make_matrix_test();
    // test_copy_matrix();
    // test_axpy_matrix();
    // test_transpose_matrix();
    // test_matmul();
    // test_activate_matrix();
    // test_gradient_matrix();
    // test_connected_layer();
    //test_matrix_speed();

    bn_forward();
    // test_batch_norm();
    printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
}

