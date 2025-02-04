#include "../headers/cuda_util.h"

/**
 * Uploads an Obj3d object on the GPU
 */
DeviceObj3d uploadObj3dToDevice(Obj3d obj) {
    DeviceObj3d d_obj;
    cudaMalloc((void**) &d_obj.pos, 3*sizeof(double));
    cudaMemcpy(d_obj.pos, obj.pos, 3*sizeof(double), cudaMemcpyDefault);
    cudaMalloc((void**) &d_obj.rot, 3*sizeof(double));
    cudaMemcpy(d_obj.rot, obj.rot, 3*sizeof(double), cudaMemcpyDefault);
    return d_obj;
}

/**
 * Uploads a Model object to the GPU
 */
DeviceModel uploadModelToDevice(Model model) {
    DeviceModel data;
    cudaMalloc((void**) &data.points, model.points.size()*sizeof(double));
    cudaMalloc((void**) &data.texCoords, model.texCoords.size()*sizeof(double));
    cudaMalloc((void**) &data.vertices, model.vertices.size()*sizeof(int));
    cudaMalloc((void**) &data.texVertices, model.texVertices.size()*sizeof(int));

    cudaMemcpy(data.points, model.points.data(), model.points.size()*sizeof(double), cudaMemcpyDefault);
    cudaMemcpy(data.texCoords, model.texCoords.data(), model.texCoords.size()*sizeof(double), cudaMemcpyDefault);
    cudaMemcpy(data.vertices, model.vertices.data(), model.vertices.size()*sizeof(int), cudaMemcpyDefault);
    cudaMemcpy(data.texVertices, model.texVertices.data(), model.texVertices.size()*sizeof(int), cudaMemcpyDefault);

    data.points_size = model.points.size();
    data.coords_size = model.texCoords.size();
    data.vertices_size = model.vertices.size();
    return data;
}

/**
 * Uploads the pixels of the given texture on the GPU
 */
Texture uploadTextureToDevice(Texture texture) {
    Texture data = texture;
    int len = texture.width*texture.height;
    cudaMalloc((void**) &data.pixels, len*sizeof(int));
    cudaMemcpy(data.pixels, texture.pixels, len*sizeof(int), cudaMemcpyDefault);
    return data;
}

/**
 * Uploads mesh data to the GPU
 */
DeviceMesh uploadMeshToDevice(Mesh mesh) {
    DeviceObj3d obj = uploadObj3dToDevice(mesh);
    return {
        obj.pos, obj.rot,
        uploadModelToDevice(mesh.model),
        uploadTextureToDevice(mesh.texture)
    };
}

/**
 * Allocates the array of pixels and the z-buffer on the GPU
 */
DeviceScreen allocateScreenDataOnDevice(DeviceScreen screen) {
    int len = screen.width*screen.height;
    cudaMalloc((void**) &screen.pixels, len*sizeof(int));
    cudaMalloc((void**) &screen.zbuffer, len*sizeof(double));
    return screen;
}

/**
 * Deallocates the Obj3d on the GPU
 */
void deleteDeviceObj3d(DeviceObj3d obj) {
    cudaFree(obj.pos);
    cudaFree(obj.rot);
}

/**
 * Deallocates the given model on the GPU
 */
void deleteDeviceModel(DeviceModel model) {
    cudaFree(model.points);
    cudaFree(model.texCoords);
    cudaFree(model.vertices);
    cudaFree(model.texVertices);
}

/**
 * Deallocates pixels of the texture on the GPU
 */
void deleteDeviceTexture(Texture texture) {
    cudaFree(texture.pixels);
}

/**
 * Deallocates mesh data on the GPU
 */
void deleteDeviceMesh(DeviceMesh mesh) {
    deleteDeviceObj3d(mesh);
    deleteDeviceModel(mesh.model);
    deleteDeviceTexture(mesh.texture);
}

/**
 * Updates the visualized pixels with the ones computed on the GPU
 */
void downloadDevicePixels(DeviceScreen screen, int* pixels) {
    int len = screen.width*screen.height;
    cudaMemcpy(pixels, screen.pixels, len*sizeof(int), cudaMemcpyDefault);
}