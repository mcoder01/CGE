#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>

#include "../headers/screen.h"
#include "../headers/world.h"
#include "../headers/camera.h"

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
        loadTexture("resources/textures/cobblestone.png", IMG_INIT_PNG)
    };
    world.addMesh(cube);

    // Initialize the screen
    screen.init();

    // Start the main loop
    bool running = true;
    while(running) {
        screen.update();

        // Catch I/O events (e.g. keyboard, mouse, etc...)
        SDL_Event event;
        if (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                running = false;
            else camera.onEvent(event);
        }
    }
}