#ifndef SCREEN_H
#define SCREEN_H

#include <SDL2/SDL.h>
#include "world.h"
#include "camera.h"

class Screen {
private:
    World* world;
    Camera* camera;
    long lastUpdate, unprocessedTime = 0;

public:
    SDL_Window* window;
    SDL_Surface* surface;
    SDL_Renderer* renderer;

    const char* title;
    int width, height, scale, resolution, fps = 0;
    double fov;

    Screen(const char*, int, int);
    void init();
    void update();
    void setWorld(World*);
    void setCamera(Camera*);
    void showFPS();
};

#endif