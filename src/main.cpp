#include<assert.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<errno.h>
#include<math.h>
#include<limits.h>
#include<chrono>
#include<iostream>

#define HEIGHT 500
#define WIDTH 500
#define PPM_SCALER 5
#define SAMPLE_SIZE 1000
#define BIAS 10.0
#define TRAIN_PASSES 100

typedef float Layer[HEIGHT][WIDTH];

static inline int clampi(int x, int low, int high)
{
    if(x < low) x = low;
    if(x > high) x = high;

    return x;
}

void layer_fill_rect(Layer layer, int x, int y, int w, int h, float value)
{
    assert(w > 0);
    assert(h > 0);

    int x0 = clampi(x, 0, WIDTH-1);
    int y0 = clampi(y, 0, HEIGHT-1);

    int x1 = clampi(x0 + w - 1, 0, WIDTH-1);
    int y1 = clampi(y0 + h - 1, 0, HEIGHT-1);

    for(int y=y0; y<=y1; y++){
        for(int x=x0; x<=x1; x++){
            layer[y][x] = value;
        }
    }
}

void layer_fill_circle(Layer layer, int cx, int cy, int r, float value)
{
    assert(r > 0);

    int x0 = clampi(cx-r, 0, WIDTH-1);
    int y0 = clampi(cy-r, 0, HEIGHT-1);

    int x1 = clampi(cx+r, 0, WIDTH-1);
    int y1 = clampi(cy+r, 0, HEIGHT-1);

    for(int y=y0; y<=y1; y++){
        for(int x=x0; x<=x1; x++){
            int dx = x - cx;
            int dy = y - cy;
            if(dx*dx + dy*dy <= r*r){
                layer[y][x] = value;
            }
        }
    }
}

void layer_save_as_ppm(Layer layer, const char* file_path)
{
    float max = layer[0][0];
    float min = layer[0][0];

    for(int y=0; y<HEIGHT; y++){
        for(int x=0; x<WIDTH; x++){
            if(layer[y][x] > max) max = layer[y][x];
            if(layer[y][x] < min) min = layer[y][x];
        }
    }

    FILE *f = fopen(file_path, "wb");

    if(f == NULL){
        fprintf(stderr, "ERROR: could not open file %s: %s\n", file_path, strerror(errno));
        exit(1);
    }
    

    fprintf(f, "P6\n%d %d 255\n", WIDTH*PPM_SCALER, HEIGHT*PPM_SCALER);

    for(int y=0; y<HEIGHT*PPM_SCALER; y++){
        for(int x=0; x<WIDTH*PPM_SCALER; x++){
            float s = (layer[y/PPM_SCALER][x/PPM_SCALER] - min) / (max - min);

            char pixel[3] = {
                (char) floor(255 * (1-s)),
                (char) floor(255 * s),
                0
            };

            fwrite(pixel, sizeof(pixel), 1, f);
        }
    }

    fclose(f);
}

void layer_save_as_bin(Layer layer, const char* file_path)
{
    FILE *f = fopen(file_path, "wb");

    if(f == NULL){
        fprintf(stderr, "ERROR: could not open file %s: %s\n", file_path, strerror(errno));
        exit(1);
    }

    fwrite(layer, sizeof(Layer), 1, f);

    fclose(f);
}

int rand_range(int low, int high)
{
    assert(low < high);
    return rand() % (high - low) + low;
}

void layer_random_rect(Layer layer)
{
    layer_fill_rect(layer, 0, 0, WIDTH, HEIGHT, 0.0f);
    int x = rand_range(0, WIDTH);
    int y = rand_range(0, HEIGHT);
    
    int w = WIDTH - x;
    if(w < 2) w = 2;
    w = rand_range(1, w);
    
    int h = HEIGHT - y;
    if(h < 2) h = 2;
    h = rand_range(1, h);

    layer_fill_rect(layer, x, y, w, h, 1.0f);
}

void layer_random_circle(Layer layer)
{
    layer_fill_rect(layer, 0, 0, WIDTH, HEIGHT, 0.0f);
    int cx = rand_range(0, WIDTH-1);
    int cy = rand_range(0, HEIGHT-1);
    int r = INT_MAX;
    if(r > cx) r = cx;
    if(r > cy) r = cy;
    if(r > WIDTH - cx) r = WIDTH - cx;
    if(r > HEIGHT - cy) r = HEIGHT - cy;

    if(r < 2) r = 2;
    r = rand_range(1, r);

    // printf("cx: %d cy: %d r: %d\n",cx, cy,  r);
    layer_fill_circle(layer, cx, cy, r, 1.0f);
}

float feed_forward(Layer inputs, Layer weights){
    float output = 0.0f;

    for(int y=0; y<HEIGHT; y++){
        for(int x=0; x<WIDTH; x++){
            output += inputs[y][x] * weights[y][x];
        }
    }

    return output;
}

void add_inputs_to_weights(Layer inputs, Layer weights)
{
    for(int y=0; y<HEIGHT; y++){
        for(int x=0; x<WIDTH; x++){
            weights[y][x] += inputs[y][x];
        }
    }
}

void sub_inputs_from_weights(Layer inputs, Layer weights)
{
    for(int y=0; y<HEIGHT; y++){
        for(int x=0; x<WIDTH; x++){
            weights[y][x] -= inputs[y][x];
        }
    }
}

int train_pass(Layer inputs, Layer weights)
{
    int adj = 0;

    for(int i=0; i<SAMPLE_SIZE; i++){
        layer_random_circle(inputs);
        if(feed_forward(inputs, weights) < BIAS){
            add_inputs_to_weights(inputs, weights);
            adj++;
        }

        layer_random_rect(inputs);
        if(feed_forward(inputs, weights) > BIAS){
            sub_inputs_from_weights(inputs, weights);
            adj++;
        }
    }

    return adj;
}

int check_pass(Layer inputs, Layer weights)
{
    int adj = 0;

    for(int i=0; i<SAMPLE_SIZE; i++){
        layer_random_rect(inputs);
        if(feed_forward(inputs, weights) > BIAS){
            adj++;
        }

        layer_random_circle(inputs);
        if(feed_forward(inputs, weights) < BIAS){
            adj++;
        }
    }

    return adj;
}

static Layer inputs;
static Layer weights;

int main()
{
    int adj, i=0;

    printf("Sample Size = %d\n", SAMPLE_SIZE);
    
    // check on untrained model
    srand(420);
    adj = check_pass(inputs, weights);
    printf("The fail rate on untrained model %g%%\n", (adj / (SAMPLE_SIZE * 2.0))*100);

    // training 
    auto start = std::chrono::high_resolution_clock::now();

    printf("Started training ...");
    do{
        i++;
        srand(69);
        adj = train_pass(inputs, weights);

        // printf("adjusted %d times\n", adj);
    }while(adj > 0);
    auto finish = std::chrono::high_resolution_clock::now();
    printf("Done in ");
    std::cout << std::chrono::duration_cast<std::chrono::seconds>(finish-start).count() << "sec\n";

    // check on trained model
    srand(420);
    adj = check_pass(inputs, weights);
    printf("The fail rate on trained model %g%%\n", (adj / (SAMPLE_SIZE * 2.0))*100);

    return 0;
}