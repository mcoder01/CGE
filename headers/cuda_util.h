#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#include "mesh.h"

typedef struct {
    double* points;
    double* texCoords;
    int* vertices;
    int* texVertices;
    int points_size;
    int coords_size;
    int vertices_size;
} DeviceModel;

typedef struct {
    double* pos;
    double* rot;
} DeviceObj3d;

typedef struct : DeviceObj3d {
    DeviceModel model;
    Texture texture;
} DeviceMesh;

typedef struct {
    int width, height;
    int scale, resolution;
    int near, far;
    int* pixels;
    double* zbuffer;
} DeviceScreen;

DeviceObj3d uploadObj3dToDevice(Obj3d);
DeviceMesh uploadMeshToDevice(Mesh);
DeviceScreen allocateScreenDataOnDevice(DeviceScreen);
void deleteDeviceObj3d(DeviceObj3d);
void deleteDeviceMesh(DeviceMesh);
void downloadDevicePixels(DeviceScreen, int*);

#endif