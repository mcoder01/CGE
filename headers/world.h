#ifndef WORLD_H
#define WORLD_H

#include "mesh.h"
#include "cuda_util.h"

class World {
private:
    std::vector<Mesh> objects;
    DeviceScreen screen;
    double viewPointTime = 0, normalComputationTime = 0;
    double faceDecompositionTime = 0, faceClippingTime = 0;
    double projectionTime = 0, areaComputationTime = 0;
    double rasterTime = 0, frameTime = 0;
    int frames = 0;

public:
    World(DeviceScreen);
    ~World();
    void addMesh(Mesh);
    void tick(double);
    void drawObjects(SDL_Surface*, Obj3d);
};

typedef struct {
    double position[3];
    double normal[3];
} Plane;

#endif