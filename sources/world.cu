#include "../headers/world.h"
#include "../headers/math_util.h"
#include "../headers/render.h"

/**
 * Performs the dot product between two vectors.
 */
__host__ __device__ double dotProduct(double* a, double* b, int dim) {
    double sum = 0;
    for (int i = 0; i < dim; i++)
        sum += a[i]*b[i];
    return sum;
}

/**
 * Performs the normalization of the given vector in-place.
 */
__host__ __device__ void normalize(double* v, int len) {
    double mag = sqrt(dotProduct(v, v, len));
    for (int i = 0; i < len; i++)
        v[i] /= mag;
}

/**
 * Check if the given vertex is in the frustum
 */
__host__ __device__ bool isInFrustum(Vertex v, Plane plane) {
    double pd = dotProduct(plane.position, plane.normal, 3);
    double vd = dotProduct(v.position, plane.normal, 3);
    return vd-pd >= 0;
}

/**
 * Returns the factor which describes the intersection point between 
 * a plane and a segment represented by its ends (vectors `a` and `b`).
 * Input vectors must be of three dimensions.
 */
__host__ __device__ double frustumIntersection(double* a, double* b, Plane plane) {
    double pd = dotProduct(plane.position, plane.normal, 3);
    double ad = dotProduct(a, plane.normal, 3);
    double bd = dotProduct(b, plane.normal, 3);
    return (pd-ad)/(bd-ad);
}

#if DEVICE == 0

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
 * Computes the normals of the viewed points according to the triangles which compose the mesh.
 * Returns a normal vector for each point of the mesh.
 */
double* computeNormals(Mesh& mesh, double* points) {
    double* normals = new double[mesh.model.points.size()]();
    std::vector<int>& indices = mesh.model.vertices;
    for (int i = 0; i < indices.size(); i += 3) {
        int pidx[3] = {indices[i]*3, indices[i+1]*3, indices[i+2]*3};
        double v1[3], v2[3];
        for (int j = 0; j < 3; j++) {
            v1[j] = points[pidx[1]+j]-points[pidx[0]+j];
            v2[j] = points[pidx[2]+j]-points[pidx[0]+j];
        }

        // Perform the cross product between two sides of the triangle
        double normal[3] = {
            v1[1]*v2[2]-v1[2]*v2[1],
            v1[2]*v2[0]-v1[0]*v2[2],
            v1[0]*v2[1]-v1[1]*v2[0]
        };

        /* Normalize the computed normal and sum it to the normals 
           related to the points involved into the computation. */
        normalize(normal, 3);
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                normals[pidx[j]+k] += normal[k];
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
        int face = -1;
        for (int j = 0; j < 3; j++) {
            int pointIdx = indices[i+j], texIdx = texIndices[i+j];
            normalize(normals+pointIdx*3, 3);
            if (face >= 0) {
                int idx = face*3+j;
                memcpy(vertices[idx].position, points+pointIdx*3, 3*sizeof(double));
                memcpy(vertices[idx].texCoord, texCoords+texIdx*2, 2*sizeof(double));
                memcpy(vertices[idx].normal, normals+pointIdx*3, 3*sizeof(double));
            } else if (dotProduct(points+pointIdx*3, normals+pointIdx*3, 3) < 0) {
                face = faces++;
                j = -1;
            }
        }
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
        double* normals = computeNormals(mesh, points);
        
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

#elif DEVICE == 1

World::World(DeviceScreen screen) {
    this->screen = allocateScreenDataOnDevice(screen);
}

void World::addMesh(Mesh mesh) {
    objects.push_back(mesh);
}

void World::tick(double deltaTime) {
    //objects[0].rot[1] += 0.1*deltaTime;
}

/**
 * Rotates and moves the points of the given mesh in the 3D space with respect
 * to the position and the rotation of the camera and the mesh object.
 */
double* viewMeshPoints(DeviceMesh& mesh, DeviceObj3d camera) {
    int num_points = mesh.model.points_size/3;
    double* viewed;

    // Rotates mesh points respect to its rotation
    cudaMalloc((void**) &viewed, mesh.model.points_size*sizeof(double));
    rotate(mesh.model.points, mesh.rot, num_points, viewed);

    // Moves mesh points according to its position and the position of the camera
    double* shift;
    cudaMalloc((void**) &shift, 3*sizeof(double));
    sub<<<3,1>>>(mesh.pos, camera.pos, 1, 3, shift);
    add<<<48,64>>>(viewed, shift, num_points, 3, viewed);
    cudaFree(shift);

    // Rotates mesh points with respect to rotation of the camera
    rotate(viewed, camera.rot, num_points, viewed);
    return viewed;
}

/**
 * Transfers the UV coordinates of the texture on the GPU
 */
double* uploadTexCoordsToDevice(DeviceMesh& mesh) {
    double* texCoords;
    cudaMalloc((void**) &texCoords, mesh.model.coords_size*sizeof(double));
    cudaMemcpy(texCoords, mesh.model.texCoords, mesh.model.coords_size*sizeof(double), cudaMemcpyDefault);
    return texCoords;
}

/**
 * Computes the normals of the viewed points according to the triangles which compose the mesh.
 * Returns a normal vector for each point of the mesh.
 */
__global__ void computeNormals(DeviceMesh& mesh, double* points, double* normals) {
    int start, size;
    distribute(mesh.model.vertices_size, 3, &start, &size);

    int* indices = mesh.model.vertices;
    for (int i = start; i < start+size; i += 3) {
        if (i >= mesh.model.vertices_size)
            return;

        int pidx[3] = {indices[i]*3, indices[i+1]*3, indices[i+2]*3};
        double v1[3], v2[3];
        for (int j = 0; j < 3; j++) {
            v1[j] = points[pidx[1]+j]-points[pidx[0]+j];
            v2[j] = points[pidx[2]+j]-points[pidx[0]+j];
        }

        // Perform the cross product between two sides of the triangle
        double normal[3] = {
            v1[1]*v2[2]-v1[2]*v2[1],
            v1[2]*v2[0]-v1[0]*v2[2],
            v1[0]*v2[1]-v1[1]*v2[0]
        };

        /* Normalize the computed normal and sum it to the normals 
           related to the points involved into the computation. */
        normalize(normal, 3);
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                atomicAdd(normals+pidx[j]+k, normal[k]);
    }
}

double* computeNormals(DeviceMesh& mesh, double* points) {
    double* normals;
    cudaMalloc((void**) &normals, mesh.model.points_size*sizeof(double));
    cudaMemset(normals, 0, mesh.model.points_size*sizeof(double));
    computeNormals<<<48,64>>>(mesh, points, normals);
    return normals;
}

/**
 * Takes the processed points, UV and normals and generates an array of vertices which represent the mesh triangles.
 * For optimization purposes, the method automatically detects invisible triangles and ignore them during the generation of the vertices.
 */
__global__ void decomposeFaces(DeviceMesh mesh, double* points, double* texCoords, double* normals, Vertex* vertices, int* len) {
    int start, size;
    distribute(mesh.model.vertices_size, 3, &start, &size);
    for (int i = start; i < start+size; i += 3) {
        if (i >= mesh.model.vertices_size)
            return;

        int face = -1;
        for (int j = 0; j < 3; j++) {
            int pointIdx = mesh.model.vertices[i+j], texIdx = mesh.model.texVertices[i+j];
            normalize(normals+pointIdx*3, 3);
            if (face >= 0) {
                int idx = face*3+j;
                memcpy(vertices[idx].position, points+pointIdx*3, 3*sizeof(double));
                memcpy(vertices[idx].texCoord, texCoords+texIdx*2, 2*sizeof(double));
                memcpy(vertices[idx].normal, normals+pointIdx*3, 3*sizeof(double));
            } else if (dotProduct(points+pointIdx*3, normals+pointIdx*3, 3) < 0) {
                face = atomicAdd(len, 1);
                j = -1;
            }
        }
    }
}

Vertex* decomposeFaces(DeviceMesh mesh, double* points, double* texCoords, double* normals, int& size) {
    Vertex* vertices;
    int* d_size;
    cudaMalloc((void**) &vertices, mesh.model.vertices_size*sizeof(Vertex));
    cudaMalloc((void**) &d_size, sizeof(int));
    cudaMemset(d_size, 0, sizeof(int));
    decomposeFaces<<<48,64>>>(mesh, points, texCoords, normals, vertices, d_size);
    cudaMemcpy(&size, d_size, sizeof(int), cudaMemcpyDefault);
    cudaFree(d_size);
    return vertices;
}

/**
 * Clips each triangle against the given plane in the 3D space. The clipped triangles
 * are stored into a the array of vertices `clipped`, while its length is stored into `len`.
 */
__global__ void clipFaces(Vertex* vertices, int faces, Plane plane, Vertex* clipped, int* len) {
    int start, size;
    distribute(faces, 1, &start, &size);
    for (int i = start; i < start+size; i++) {
        if (i >= faces) break;

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
            int index = atomicAdd(len, 1)*3;
            double t1 = frustumIntersection(inside[0].position, outside[0].position, plane);
            double t2 = frustumIntersection(inside[0].position, outside[1].position, plane);
            clipped[index] = inside[0];
            clipped[index+1] = lerpVertex(&inside[0], &outside[0], t1);
            clipped[index+2] = lerpVertex(&inside[0], &outside[1], t2);
        } else if (in_num == 2) {
            int idx1 = atomicAdd(len, 1)*3;
            double t1 = frustumIntersection(inside[0].position, outside[0].position, plane);
            double t2 = frustumIntersection(inside[1].position, outside[0].position, plane);

            clipped[idx1] = inside[0];
            clipped[idx1+1] = inside[1];
            clipped[idx1+2] = lerpVertex(&inside[0], &outside[0], t1);

            int idx2 = atomicAdd(len, 1)*3;
            clipped[idx2] = inside[1];
            clipped[idx2+1] = clipped[idx1+2];
            clipped[idx2+2] = lerpVertex(&inside[1], &outside[0], t2);
        } else if (in_num == 3) {
            // The triangle should not be clipped
            int index = atomicAdd(len, 1)*3;
            memcpy(clipped+index, inside, 3*sizeof(Vertex));
        }
    }
}

Vertex* clipFacesAgainstPlane(Vertex* vertices, int& faces, Plane plane) {
    Vertex* clipped;
    int* output_size;
    cudaMalloc((void**) &clipped, 6*faces*sizeof(Vertex));
    cudaMalloc((void**) &output_size, sizeof(int));
    cudaMemset(output_size, 0, sizeof(int));
    clipFaces<<<48,64>>>(vertices, faces, plane, clipped, output_size);
    cudaMemcpy(&faces, output_size, sizeof(int), cudaMemcpyDefault);
    cudaFree(output_size);
    return clipped;
}

/**
 * Clips the given triangles against the near and far planes of the frustum.
 */
Vertex* clipFaces(Vertex* vertices, int& faces) {
    Vertex* nearClipped = clipFacesAgainstPlane(vertices, faces, {0, 0, -1, 0, 0, -1});
    Vertex* clipped = clipFacesAgainstPlane(nearClipped, faces, {0, 0, -100, 0, 0, 1});
    cudaFree(nearClipped);
    return clipped;
}

/**
 * Perform the perspective projection of the given vertices.
 */
__global__ void project(Vertex* vertices, int num_points, DeviceScreen screen) {
    int start, size;
    distribute(num_points, 1, &start, &size);
    for (int i = start; i < start+size; i++) {
        if (i >= num_points)
            return;

        double absZ = abs(vertices[i].position[2]);
        vertices[i].projection[0] = vertices[i].position[0]/absZ*screen.scale+screen.width/2;
        vertices[i].projection[1] = -vertices[i].position[1]/absZ*screen.scale+screen.height/2;
    }
}

/**
 * This method finds the smallest diagonal (described by the top left and bottom right corners) of
 * the rectangle which contains the projections of all the given vertices.
 */
__global__ void findDiagonal(Vertex* vertices, int num_points, DeviceScreen screen, int* diagonal) {
    int start, size;
    distribute(num_points, 1, &start, &size);

    int minX = INT_MAX, minY = INT_MAX;
    int maxX = 0, maxY = 0;
    for (int i = start; i < start+size; i++) {
        if (i >= num_points)
            break;

        int x = min(max((int) vertices[i].projection[0], 0), screen.width-1);
        int y = min(max((int) vertices[i].projection[1], 0), screen.height-1);

        minX = min(minX, x);
        maxX = max(maxX, x);
        minY = min(minY, y);
        maxY = max(maxY, y);
    }

    if (start < num_points) {
        atomicMin(&diagonal[0], minX);
        atomicMin(&diagonal[1], minY);
        atomicMax(&diagonal[2], maxX);
        atomicMax(&diagonal[3], maxY);
    }
}

/**
 * Returns the rectangle which describe the smallest area containing the projections
 * of all the vertices
 */
SDL_Rect computeRenderArea(Vertex* vertices, int num_points, DeviceScreen screen) {
    int diagonal[4] = {INT_MAX, INT_MAX, 0, 0};
    int* d_diagonal;
    cudaMalloc((void**) &d_diagonal, sizeof(diagonal));
    cudaMemcpy(d_diagonal, diagonal, sizeof(diagonal), cudaMemcpyDefault);
    findDiagonal<<<48,64>>>(vertices, num_points, screen, d_diagonal);
    cudaMemcpy(diagonal, d_diagonal, sizeof(diagonal), cudaMemcpyDefault);
    cudaFree(d_diagonal);
    return {diagonal[0], diagonal[1], diagonal[2]-diagonal[0]+1, diagonal[3]-diagonal[1]+1};
}

/**
 * This method is called each frame to process and render all the meshes of the scene.
 */
void World::drawObjects(SDL_Surface* surface, Obj3d camera) {
    initScreen<<<8,224>>>(screen); // Clean the screen

    DeviceObj3d d_camera = uploadObj3dToDevice(camera);
    for (Mesh mesh : objects) {
        // Compute all the data required to render the mesh
        DeviceMesh d_mesh = uploadMeshToDevice(mesh);
        double* points = viewMeshPoints(d_mesh, d_camera);
        double* texCoords = uploadTexCoordsToDevice(d_mesh);
        double* normals = computeNormals(d_mesh, points);
        
        // Decompose points into triangles and clip them
        int faces;
        Vertex* vertices = decomposeFaces(d_mesh, points, texCoords, normals, faces);
        Vertex* clipped = clipFaces(vertices, faces);
        cudaFree(vertices);

        /* Project the final vertices, compute the rendering area 
         * of the screen and fit the texture on the mesh */
        project<<<48,64>>>(clipped, faces*3, screen);
        SDL_Rect area = computeRenderArea(clipped, faces*3, screen);
        raster<<<48,64>>>(clipped, faces*3, d_mesh.texture, screen, area);

        // Free all the memory allocated on the GPU for this mesh
        deleteDeviceMesh(d_mesh);
        cudaFree(points);
        cudaFree(texCoords);
        cudaFree(normals);
        cudaFree(clipped);
    }

    deleteDeviceObj3d(d_camera);
    downloadDevicePixels(screen, (int*) surface->pixels); // Update pixels
}

#endif