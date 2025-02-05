#include "../headers/camera.h"

Camera::Camera(DeviceScreen screen, double x, double y, double z):screen(screen) {
    pos[0] = x;
    pos[1] = y;
    pos[2] = z;
    memset(rot, 0, 3*sizeof(double));
}

/**
 * Updates the rotation angles of the camera
 */
void Camera::rotate(int dx, int dy) {
    rot[0] -= (double) dy/screen.height;
    rot[1] -= (double) dx/screen.width;
}

/**
 * Set the direction where the camera has to move
 */
void Camera::move(SDL_Scancode scancode) {
    switch(scancode) {
        case SDL_SCANCODE_W:
            dz = -1;
            break;
        case SDL_SCANCODE_A:
            dx = -1;
            break;
        case SDL_SCANCODE_S:
            dz = 1;
            break;
        case SDL_SCANCODE_D:
            dx = 1;
            break;
        case SDL_SCANCODE_LSHIFT:
            dy = -1;
            break;
        case SDL_SCANCODE_SPACE:
            dy = 1;
            break;
    }
}

/**
 * Stops the movement of the camera, when
 * the corresponing key on the keyboard is released
 */
void Camera::stop(SDL_Scancode code) {
    if (code == SDL_SCANCODE_W || code == SDL_SCANCODE_S) dz = 0;
    else if (code == SDL_SCANCODE_A || code == SDL_SCANCODE_D) dx = 0;
    else if (code == SDL_SCANCODE_LSHIFT || code == SDL_SCANCODE_SPACE) dy = 0;
}

/**
 * Check if the camera has to rotate or to move into the space
 */
void Camera::onEvent(SDL_Event event) {
    switch (event.type) {
        case SDL_MOUSEMOTION:
            if (event.motion.state == SDL_BUTTON_LMASK)
                rotate(event.motion.xrel, event.motion.yrel);
            break;
        case SDL_KEYDOWN:
            move(SDL_GetScancodeFromKey(event.key.keysym.sym));
            break;
        case SDL_KEYUP:
            stop(SDL_GetScancodeFromKey(event.key.keysym.sym));
            break;
    }
}

/**
 * This method is called every frame for updating the camera
 */
void Camera::tick(double deltaTime) {
    pos[0] += (dx*cos(rot[1])+dz*sin(rot[1]))*moveSpeed*deltaTime;
    pos[1] += dy*moveSpeed*deltaTime;
    pos[2] += (dz*cos(rot[1])-dx*sin(rot[1]))*moveSpeed*deltaTime;
}