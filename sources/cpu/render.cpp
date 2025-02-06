/**
 * Fills the screen pixels by fitting the given texture on the 2D space 
 * covered by the polygons stored into the `vertices` array.
 */
void raster(Vertex* vertices, int size, Texture texture, DeviceScreen screen, SDL_Rect area) {
    int res = screen.resolution;
    int scaledW = area.w/res;
    int scaledH = area.h/res;
    for (int scaledX = 0; scaledX < scaledW; scaledX++)
        for (int scaledY = 0; scaledY < scaledH; scaledY++) {
            int x = scaledX*res+area.x;
            int y = scaledY*res+area.y;
            int index = y*screen.width+x;
            bool validPixel = false;
            for (int j = 0; j < size; j += 3) {
                Vertex triangle[3] = {vertices[j], vertices[j+1], vertices[j+2]};
                Vertex vertex;
                double z;
                if (computeInnerVertex(triangle, x, y, &vertex, &z) && (screen.zbuffer[index] == 0 || z > screen.zbuffer[index])) {
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