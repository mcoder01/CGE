#ifndef MESH_H
#define MESH_H

#include <vector>
#include <SDL2/SDL.h>

typedef struct {
    std::vector<double> points;
    std::vector<double> texCoords;
    std::vector<int> vertices;
    std::vector<int> texVertices;
} Model;

typedef struct {
    int width;
    int height;
    int* pixels = NULL;
    SDL_Surface* image = NULL;
} Texture;

typedef struct {
    double pos[3];
    double rot[3];
} Obj3d;

typedef struct : Obj3d {
    Model model;
    Texture texture;
} Mesh;

Model loadModel(const char*);
Texture loadTexture(const char*, int);

#endif