#include "../headers/render.h"
#include "../headers/math_util.h"
#include <cfloat>

template <typename T>
__host__ __device__ void swap(T* a, T* b) {
    T temp = *a;
    *a = *b;
    *b = temp;
}

/**
 * Swaps two vertices to sort them in ascending order according to Y coordinates.
 */
__host__ __device__ void compswap(Vertex* vertices, int i, int j) {
    if (vertices[i].projection[1] > vertices[j].projection[1])
        swap(&vertices[i], &vertices[j]);
}

/**
 * Multiplies the components of the given vector with the value stored into `val`.
 * Stores the result into `result` array.
 */
__host__ __device__ void scaleVector(double* v, double val, int dim, double* result) {
    for (int i = 0; i < dim; i++)
        result[i] = v[i]*val;
}

/**
 * Scales a vertex by the given value, through scaling each constituent vector
 */
__host__ __device__ void scaleVertex(Vertex* vertex, double value, Vertex* result) {
    scaleVector(vertex->position, value, 3, result->position);
    scaleVector(vertex->texCoord, value, 2, result->texCoord);
    scaleVector(vertex->normal, value, 3, result->normal);
    scaleVector(vertex->projection, value, 2, result->projection);
}

/**
 * Computes a linear interpolation of the vertices on the given side
 * with respect to the parameter `t` and stores the resulting vertex,
 * into `result`.
 */
__host__ __device__ double lerpOnY(Side side, double t, Vertex* result) {
    double zbegin = 1/side.v1->position[2];
    double zend = 1/side.v2->position[2];
    Vertex top, bottom;
    scaleVertex(side.v1, zbegin, &top);
    scaleVertex(side.v2, zend, &bottom);
    *result = lerpVertex(&top, &bottom, t);
    return lerp(zbegin, zend, t);
}

/**
 * Computes the interpolation of the given triangle vertices according to the given point on the screen
 * in order to generate a new vertex which represents this point in the 3D space. 
 * Stores the interpolated vertex into `inner` and the corresponding z coordinate into `z`.
 */
__host__ __device__ bool computeInnerVertex(Vertex* triangle, int x, int y, Vertex* inner, double* z) {
    // Sorts the vertices of the triangle with respect to the projected Y position.
    compswap(triangle, 0, 1);
    compswap(triangle, 0, 2);
    compswap(triangle, 1, 2);

    int y1 = triangle[0].projection[1];
    int y2 = triangle[1].projection[1];
    int y3 = triangle[2].projection[1];

    // Check if the given Y coordinate is inside the triangle surface
    if (y < y1 || y >= y3) return false;

    Side leftSide, rightSide = {&triangle[0], &triangle[2]};
    double left_ty, right_ty = (double) (y-y1)/(y3-y1);

    if (y < y2) {
        left_ty = (double) (y-y1)/(y2-y1);
        leftSide = {&triangle[0], &triangle[1]};
    } else {
        left_ty = (double) (y-y2)/(y3-y2);
        leftSide = {&triangle[1], &triangle[2]};
    }

    // Interpolates the X coordinates for the ends of the segment of the surface of the triangle with respect to the y coordinate
    int startX = lerp(leftSide.v1->projection[0], leftSide.v2->projection[0], left_ty);
    int endX = lerp(rightSide.v1->projection[0], rightSide.v2->projection[0], right_ty);

    if (startX > endX) {
        // Swaps the left and right side if the first X is greater than the last X
        swap(&left_ty, &right_ty);
        swap(&leftSide, &rightSide);
        swap(&startX, &endX);
    }

    // Checks if the given X coordinate is inside the triangle surface
    if (x < startX || x >= endX) return false;
    
    // Performs the interpolation of the inner vertex
    Vertex left, right;
    double zleft = lerpOnY(leftSide, left_ty, &left);
    double zright = lerpOnY(rightSide, right_ty, &right);
    double tx = (double) (x-startX)/(endX-startX);
    *z = 1/lerp(zleft, zright, tx);
    *inner = lerpVertex(&left, &right, tx);
    return true;
}

/**
 * Fills the screen pixels by fitting the given texture on the 2D space 
 * covered by the polygons stored into the `vertices` array.
 * Each CUDA thread is responsible of a specific portion of screen pixels.
 */
__global__ void raster(Vertex* vertices, int size, Texture texture, DeviceScreen screen, SDL_Rect* area) {
    int res = screen.resolution;
    int scaledW = area->w/res;
    int scaledH = area->h/res;
    int start, num_pixels;
    distribute(scaledW*scaledH, 1, &start, &num_pixels);
    
    for (int i = start; i < start+num_pixels; i++) {
        if (i >= scaledW*scaledH)
            break;
 
        int x = i%scaledW*res+area->x;
        int y = i/scaledW*res+area->y;
        int index = y*screen.width+x;
        bool validPixel = false;
        for (int j = 0; j < size; j += 3) {
            Vertex triangle[3] = {vertices[j], vertices[j+1], vertices[j+2]};
            Vertex vertex;
            double z;
            if (computeInnerVertex(triangle, x, y, &vertex, &z) && z > screen.zbuffer[index]) {
                int u = min((int) (vertex.texCoord[0]*z*texture.width), texture.width-1);
                int v = min((int) (vertex.texCoord[1]*z*texture.height), texture.height-1);
                int rgb = texture.pixels ? texture.pixels[v*texture.width+u] : 0xffffff;
                //double[] point = ArrayUtil.scale(vertex.position(), z);
                screen.pixels[index] = rgb;
                screen.zbuffer[index] = z;
                validPixel = true;
            }
        }
        
        if (validPixel)
            for (int i = y; i < y+res; i++)
                for (int j = x; j < x+res; j++)
                    screen.pixels[i*screen.width+j] = screen.pixels[index];
    }
}

/**
 * Resets the pixels and the z-buffer of the screen
 */
__global__ void initScreen(DeviceScreen screen) {
    int start, num_pixels;
    distribute(screen.width*screen.height, 1, &start, &num_pixels);
    
    for (int index = start; index < start+num_pixels; index++)
        if (index < screen.width*screen.height) {
            screen.pixels[index] = 0x000000;
            screen.zbuffer[index] = -DBL_MAX;
        }
}