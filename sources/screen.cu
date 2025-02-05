#include "../headers/screen.h"
#include "../headers/math_util.h"

#include <chrono>
#include <cstdio>

Screen::Screen(const char* title, int width, int height):title(title),width(width),height(height),fov(M_PI/4),scale(height),resolution(1) {
    // Creates the window
    SDL_Init(SDL_INIT_VIDEO);
    window = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, 0);
    renderer = SDL_CreateRenderer(window, -1, 0);
    surface = SDL_GetWindowSurface(window);
}

long nanoTime() {
    std::chrono::_V2::steady_clock::time_point time = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count();
}

void Screen::init() {
    lastUpdate = nanoTime();
}

/**
 * This method is called on each frame to update
 * objects on the screen, like the camera and the world.
 * It also counts the number of FPS (Frame-Per-Second).
 */
void Screen::update() {
    long now = nanoTime();
    long passedTime = now-lastUpdate;
    unprocessedTime += passedTime;
    lastUpdate = now;

    fps++;
    if (unprocessedTime >= 1E9) {
        showFPS();
        unprocessedTime -= 1E9;
        fps = 0;
    }

    // Update the camera and the world
    double deltaTime = passedTime/1E9;
    camera->tick(deltaTime);
    world->tick(deltaTime);

    // Render the scene captured by the camera on the screen
    world->drawObjects(surface, *camera);
    SDL_UpdateWindowSurface(window);
}

void Screen::setWorld(World* world) {
    this->world = world;
}

void Screen::setCamera(Camera* camera) {
    this->camera = camera;
}

/**
 * Shows the FPS value on the title bar of the window
 */
void Screen::showFPS() {
    char buffer[100];
    sprintf(buffer, "%s - FPS: %d", title, fps);
    SDL_SetWindowTitle(window, buffer);
}