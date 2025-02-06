#include <thread>
#include <atomic>
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>

#include "../headers/screen.h"
#include "../headers/camera.h"
#include "../headers/world.h"

std::atomic running(true);

/**
 * Catch I/O events (e.g. keyboard, mouse, etc...)
 */
void catchEvents(Camera& camera) {
    while(running) {
        SDL_Event event;
        if (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                running = false;
            else camera.onEvent(event);
        }
    }
}

int main() {
    // Opens a new window and creates the corresponding screen object
    Screen screen("CUDA Graphic Engine", 1280, 720);

    // Creates the camera and the world which will contain all the objects of the scene
    Camera camera({screen.width, screen.height}, 0, 0, 3);
    World world({screen.width, screen.height, screen.scale, screen.resolution});

    screen.setWorld(&world);
    screen.setCamera(&camera);

    // Add a cube into the world
    Mesh cube = {
        {0},
        loadModel("resources/models/cube.obj"),
        loadTexture("resources/textures/log.png", IMG_INIT_PNG)
    };
    world.addMesh(cube);

    // Initialize the screen
    screen.init();

    // Start the main loop and event listener
    std::thread loop(catchEvents, std::ref(camera));
    while(running)
        screen.update();
    loop.join();
}