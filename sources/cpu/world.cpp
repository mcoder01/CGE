World::World(DeviceScreen screen):screen(screen) {}

void World::addMesh(Mesh mesh) {
    objects.push_back(mesh);
}

void World::tick(double deltaTime) {
    objects[0].rot[1] += 0.1*deltaTime;
}

/**
 * Rotates and moves the points of the given mesh in the 3D space with respect
 * to the position and the rotation of the camera and the mesh object.
 */
double* viewMeshPoints(Mesh& mesh, Obj3d& camera) {
    int num_points = mesh.model.points.size()/3;
    double* viewed = new double[mesh.model.points.size()];

    // Rotates mesh points respect to its rotation
    rotate(mesh.model.points.data(), mesh.rot, num_points, viewed);

    // Moves mesh points according to its position and the position of the camera
    double shift[3];
    sub(mesh.pos, camera.pos, 1, 3, shift);
    add(viewed, shift, num_points, 3, viewed);

    // Rotates mesh points with respect to rotation of the camera
    rotate(viewed, camera.rot, num_points, viewed);
    return viewed;
}

/**
 * Computes the normal of the triangle determined by the given point indices.
 */
double* computeNormal(double* points, int* indices) {
    int pidx[3] = {indices[0]*3, indices[1]*3, indices[2]*3};
    double v1[3], v2[3];
    for (int i = 0; i < 3; i++) {
        v1[i] = points[pidx[1]+i]-points[pidx[0]+i];
        v2[i] = points[pidx[2]+i]-points[pidx[0]+i];
    }

    // Perform the cross product between two sides of the triangle
    double* normal = new double[3] {
        v1[1]*v2[2]-v1[2]*v2[1],
        v1[2]*v2[0]-v1[0]*v2[2],
        v1[0]*v2[1]-v1[1]*v2[0]
    };

    normalize(normal, 3);
    return normal;
}

/**
 * Computes the normal for each triangle of the mesh and sums them
 * in order to obtain a smoother version, which can be used for shading.
 */
double* computeAggregatedNormals(Mesh& mesh, double* points) {
    double* normals = new double[mesh.model.points.size()];
    std::vector<int>& indices = mesh.model.vertices;
    for (int i = 0; i < indices.size(); i += 3) {
        double* normal = computeNormal(points, &indices[i]);
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                normals[indices[i+j]+k] += normal[k];
        delete[] normal;
    }

    return normals;
}

/**
 * Takes the processed points, UV and normals and generates an array of vertices which represent the mesh triangles.
 * For optimization purposes, the method automatically detects invisible triangles and ignore them during the generation of the vertices.
 */
int decomposeFaces(Mesh& mesh, double* points, double* texCoords, double* normals, Vertex* vertices) {
    std::vector<int>& indices = mesh.model.vertices;
    std::vector<int>& texIndices = mesh.model.texVertices;
    int faces = 0;
    for (int i = 0; i < indices.size(); i += 3) {
        double* normal = computeNormal(points, &indices[i]);
        bool visible = false;
        for (int j = 0; j < 3; j++) {
            int pointIdx = indices[i+j], texIdx = texIndices[i+j];
            if (visible) {
                int idx = (faces-1)*3+j;
                memcpy(vertices[idx].position, points+pointIdx*3, 3*sizeof(double));
                memcpy(vertices[idx].texCoord, texCoords+texIdx*2, 2*sizeof(double));
                memcpy(vertices[idx].normal, normals+pointIdx*3, 3*sizeof(double));
            } else if (dotProduct(points+pointIdx*3, normal, 3) < 0) {
                visible = true;
                faces++;
                j = -1;
            }
        }

        delete[] normal;
    }

    return faces;
}

/**
 * Clips each triangle against the given plane in the 3D space. The clipped triangles
 * are stored into a the array of vertices `clipped`, while its length is stored into `len`.
 */
void clipFaces(Vertex* vertices, int faces, Plane plane, Vertex* clipped, int* len) {
    *len = 0;
    for (int i = 0; i < faces; i++) {
        // For each triangle, divide points inside the frustum from outside ones
        Vertex inside[3], outside[3];
        int in_num = 0, out_num = 0;
        for (int j = 0; j < 3; j++) {
            Vertex v = vertices[i*3+j];
            if (isInFrustum(v, plane))
                inside[in_num++] = v;
            else outside[out_num++] = v;
        }

        // Clip the triangle relying on the number of inside and outside points
        if (in_num == 1) {
            int index = ((*len)++)*3;
            double t1 = frustumIntersection(inside[0].position, outside[0].position, plane);
            double t2 = frustumIntersection(inside[0].position, outside[1].position, plane);
            clipped[index] = inside[0];
            clipped[index+1] = lerpVertex(&inside[0], &outside[0], t1);
            clipped[index+2] = lerpVertex(&inside[0], &outside[1], t2);
        } else if (in_num == 2) {
            int idx1 = ((*len)++)*3;
            double t1 = frustumIntersection(inside[0].position, outside[0].position, plane);
            double t2 = frustumIntersection(inside[1].position, outside[0].position, plane);

            clipped[idx1] = inside[0];
            clipped[idx1+1] = inside[1];
            clipped[idx1+2] = lerpVertex(&inside[0], &outside[0], t1);

            int idx2 = ((*len)++)*3;
            clipped[idx2] = inside[1];
            clipped[idx2+1] = clipped[idx1+2];
            clipped[idx2+2] = lerpVertex(&inside[1], &outside[0], t2);
        } else if (in_num == 3) {
            // The triangle should not be clipped
            int index = ((*len)++)*3;
            memcpy(clipped+index, inside, 3*sizeof(Vertex));
        }
    }
}

/**
 * Clips the given triangles against the near and far planes of the frustum.
 */
Vertex* clipFaces(Vertex* vertices, int& faces) {
    Vertex nearClipped[faces*6];
    clipFaces(vertices, faces, {0, 0, -1, 0, 0, -1}, nearClipped, &faces);
    Vertex* clipped = new Vertex[faces*6];
    clipFaces(nearClipped, faces, {0, 0, -100, 0, 0, 1}, clipped, &faces);
    return clipped;
}

/**
 * Perform the perspective projection of the given vertices.
 */
void project(Vertex* vertices, int num_points, DeviceScreen screen) {
    for (int i = 0; i < num_points; i++) {
        double absZ = abs(vertices[i].position[2]);
        vertices[i].projection[0] = vertices[i].position[0]/absZ*screen.scale+screen.width/2;
        vertices[i].projection[1] = -vertices[i].position[1]/absZ*screen.scale+screen.height/2;
    }
}

/**
 * Returns the rectangle which describe the smallest area containing the projections
 * of all the vertices
 */
SDL_Rect computeRenderArea(Vertex* vertices, int num_points, DeviceScreen screen) {
    int diagonal[4] = {INT_MAX, INT_MAX, 0, 0};
    for (int i = 0; i < num_points; i++) {
        int x = min(max((int) vertices[i].projection[0], 0), screen.width-1);
        int y = min(max((int) vertices[i].projection[1], 0), screen.height-1);

        diagonal[0] = min(diagonal[0], x);
        diagonal[1] = min(diagonal[1], y);
        diagonal[2] = max(diagonal[2], x);
        diagonal[3] = max(diagonal[3], y);
    }

    return {diagonal[0], diagonal[1], diagonal[2]-diagonal[0], diagonal[3]-diagonal[1]};
}

/**
 * This method is called each frame to process and render all the meshes of the scene.
 */
void World::drawObjects(SDL_Surface* surface, Obj3d camera) {
    screen.pixels = (int*) surface->pixels;
    screen.zbuffer = new double[screen.width*screen.height]();

    SDL_Rect screenRect = {0, 0, screen.width, screen.height};
    SDL_FillRect(surface, &screenRect, 0x000000);

    for (Mesh mesh : objects) {
        // Compute all the data required to render the mesh
        double* points = viewMeshPoints(mesh, camera);
        double* normals = computeAggregatedNormals(mesh, points);
        
        // Decompose points into triangles and clip them
        Vertex vertices[mesh.model.vertices.size()];
        int faces = decomposeFaces(mesh, points, mesh.model.texCoords.data(), normals, vertices);
        Vertex* clipped = clipFaces(vertices, faces);

        /* Project the final vertices, compute the rendering area 
         * of the screen and fit the texture on the mesh */
        project(clipped, faces*3, screen);
        SDL_Rect area = computeRenderArea(clipped, faces*3, screen);
        raster(clipped, faces*3, mesh.texture, screen, area);

        // Free all the memory allocated on the GPU for this mesh
        delete[] points;
        delete[] normals;
        delete[] clipped;
    }

    delete[] screen.zbuffer;
}