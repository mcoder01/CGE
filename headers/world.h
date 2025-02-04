#ifndef WORLD_H
#define WORLD_H

#include "mesh.h"
#include "cuda_util.h"
#include "render.h"

class World {
private:
    std::vector<Mesh> objects;
    DeviceScreen screen;

public:
    World(DeviceScreen);
    void addMesh(Mesh);
    void tick(double);
    void drawObjects(Obj3d, SDL_Renderer*, int*);
};

typedef struct {
    double position[3];
    double normal[3];
} Plane;

#endif