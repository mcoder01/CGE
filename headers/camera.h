#ifndef CAMERA_H
#define CAMERA_H

#include "cuda_util.h"
#include "mesh.h"

class Camera : public Obj3d {
private:
    DeviceScreen screen;
    double moveSpeed = 5;
    int dx = 0, dy = 0, dz = 0;

    void rotate(int, int);
    void move(SDL_Scancode);
    void stop(SDL_Scancode);

public:
    Camera(DeviceScreen, double x, double y, double z);
    void onEvent(SDL_Event);
    void tick(double);
};

#endif