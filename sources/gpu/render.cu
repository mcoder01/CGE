/**
 * Fills the screen pixels by fitting the given texture on the 2D space 
 * covered by the polygons stored into the `vertices` array.
 * Each CUDA thread is responsible of a specific portion of screen pixels.
 */
__global__ void raster(Vertex* vertices, int size, Texture texture, DeviceScreen screen, SDL_Rect area) {
    int res = screen.resolution;
    int scaledW = area.w/res;
    int scaledH = area.h/res;
    int start, num_pixels;
    distribute(scaledW*scaledH, 1, &start, &num_pixels);
    
    for (int i = start; i < start+num_pixels; i++) {
        if (i >= scaledW*scaledH)
            break;
 
        int x = i%scaledW*res+area.x;
        int y = i/scaledW*res+area.y;
        int index = y*screen.width+x;
        bool validPixel = false;
        for (int j = 0; j < size; j += 3) {
            Vertex triangle[3] = {vertices[j], vertices[j+1], vertices[j+2]};
            Vertex vertex;
            double z;
            if (computeInnerVertex(triangle, x, y, &vertex, &z) && z > screen.zbuffer[index]) {
                int u = min((int) (vertex.texCoord[0]*z*texture.width), texture.width-1);
                int v = min((int) (vertex.texCoord[1]*z*texture.height), texture.height-1);
                int rgb = texture.pixels ? texture.pixels[v*texture.width+u] : 0xffffff;
                screen.pixels[index] = rgb;
                screen.zbuffer[index] = z;
                validPixel = true;
            }
        }
        
        if (validPixel)
            for (int i = y; i < y+res; i++)
                for (int j = x; j < x+res; j++)
                    screen.pixels[i*screen.width+j] = screen.pixels[index];
    }
}

/**
 * Resets the pixels and the z-buffer of the screen
 */
__global__ void initScreen(DeviceScreen screen) {
    int start, num_pixels;
    distribute(screen.width*screen.height, 1, &start, &num_pixels);
    
    for (int index = start; index < start+num_pixels; index++)
        if (index < screen.width*screen.height) {
            screen.pixels[index] = 0x000000;
            screen.zbuffer[index] = -DBL_MAX;
        }
}