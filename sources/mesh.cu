#include "../headers/mesh.h"

#include <string>
#include <fstream>
#include <sstream>
#include <SDL2/SDL_image.h>

/**
 * Removes spaces from the beginning and the end of the given string.
 */
std::string trim(std::string& s) {
    int start = 0, end = s.size();
    while (start < end && std::isspace(s[start]))
        start++;

    while (end > start && std::isspace(s[end-1]))
        end--;

    return s.substr(start, end-start);
}

/**
 * Splits the given string into a vector of substrings, 
 * each delimited by the `delim` character.
 */
std::vector<std::string> split(std::string line, char delim) {
    std::stringstream ss(line);
    std::vector<std::string> data;
    std::string token;
    while (std::getline(ss, token, delim)) {
        token = trim(token);
        if (token != "")
            data.push_back(token);
    }
    return data;
}

/**
 * Loads a vector of the given dimension into the `output` vector.
 */
void loadVector(std::vector<std::string> info, int dim, std::vector<double>& output) {
    for (int i = 0; i < dim; i++)
        output.push_back(stod(info[i+1]));
}

/**
 * Loads face(s) from the given vector of strings into the given `Model` object.
 * The `shift` parameter is used to split quadrilater faces into triangles.
 */
void loadFace(std::vector<std::string> info, Model& model, int shift) {
    int defaultTexIndices[3] = {3, shift, 1+shift};
    int indices[3] = {1, 2+shift, 3+shift};
    for (int i = 0; i < 3; i++) {
        std::vector<std::string> vertices = split(info[indices[i]], '/');
        model.vertices.push_back(stoi(vertices[0])-1);
        if (vertices.size() >= 2)
            model.texVertices.push_back(stoi(vertices[1])-1);
        else model.texVertices.push_back(defaultTexIndices[i]);
    }
}

/**
 * Loads the object model from the file in the specified `filepath`.
 */
Model loadModel(const char* filepath) {
    Model model;
    std::ifstream is(filepath);
    while(!is.eof()) {
        std::string line;
        std::getline(is, line, '\n');
        std::vector<std::string> info = split(line, ' ');
        if (info.size() == 0) continue;
        if (info[0] == "v") loadVector(info, 3, model.points);
        else if (info[0] == "vt") loadVector(info, 2, model.texCoords);
        else if (info[0] == "f") {
            loadFace(info, model, 0);
            if (info.size() == 5)
                loadFace(info, model, 1);
        }
    }

    if (model.texCoords.empty()) {
        int coords[8] = {
            0, 0,
            1, 0,
            1, 1,
            0, 1
        };
        model.texCoords = std::vector<double>(coords, coords+8);
    }

    return model;
}

/**
 * Loads an image from the specified `filepath` into
 * a new instance of `Texture` and then returns it.
 */
Texture loadTexture(const char* filepath, int img_type) {
    IMG_Init(img_type);
    SDL_Surface* surface = IMG_Load(filepath);
    SDL_Surface* newSurface = SDL_ConvertSurfaceFormat(surface, SDL_PIXELFORMAT_RGB888, 0);
    delete surface;
    return {
        surface->w,
        surface->h,
        (int*) newSurface->pixels,
        newSurface
    };
}